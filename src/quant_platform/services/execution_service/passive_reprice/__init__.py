"""Passive limit cancel/replace coordination."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import (
    CancelReplaceRequest,
    ExecutionTactic,
    OrderIntent,
    OrderSide,
    OrderType,
    VenueRoute,
)
from quant_platform.services.execution_service.passive_reprice.passive_reprice_evidence import (
    append_passive_reprice_evidence,
)
from quant_platform.services.execution_service.passive_reprice.passive_reprice_models import (
    PassiveReplacementFactory,
    PassiveRepriceAction,
    PassiveRepriceBroker,
    PassiveRepriceDecision,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.core.contracts import (
        Clock,
        ExecutionRouter,
        OrderRepository,
        OrderStateStore,
    )
    from quant_platform.core.domain.orders import BrokerOrder
    from quant_platform.core.domain.production import ExecutionTacticPolicy

log = structlog.get_logger(__name__)


class PassiveLimitRepriceCoordinator:
    """Cancel stale passive limits and optionally place caller-built replacements.

    The coordinator does not construct replacement orders itself because that
    would bypass cash/risk ownership.  Callers that can safely reserve cash and
    preserve attribution may provide a replacement factory; otherwise due
    passive orders are cancelled and recorded as such.
    """

    def __init__(
        self,
        *,
        policy: ExecutionTacticPolicy,
        router: ExecutionRouter,
        broker: PassiveRepriceBroker,
        order_repo: OrderRepository,
        clock: Clock,
        reference_price_lookup: Callable[[OrderIntent], Decimal | None],
        order_state: OrderStateStore | None = None,
        replacement_factory: PassiveReplacementFactory | None = None,
    ) -> None:
        self._policy = policy
        self._router = router
        self._broker = broker
        self._order_repo = order_repo
        self._clock = clock
        self._reference_price_lookup = reference_price_lookup
        self._order_state = order_state
        self._replacement_factory = replacement_factory

    async def run_once(self) -> list[PassiveRepriceDecision]:
        """Evaluate currently open broker orders once."""
        decisions: list[PassiveRepriceDecision] = []
        if not self._policy.passive_limit_enabled:
            return decisions

        for broker_order in await self._broker.fetch_open_orders():
            decision = await self._evaluate_open_order(broker_order)
            decisions.append(decision)
            await append_passive_reprice_evidence(
                order_state=self._order_state,
                clock=self._clock,
                decision=decision,
            )
        return decisions

    async def _evaluate_open_order(self, broker_order: BrokerOrder) -> PassiveRepriceDecision:
        if broker_order.broker_order_id is None:
            return self._skip(broker_order.order_id, "missing_broker_order_id")

        intent = await self._order_repo.get_intent(broker_order.order_id)
        if intent is None:
            return self._skip(
                broker_order.order_id,
                "missing_order_intent",
                broker_order_id=broker_order.broker_order_id,
            )
        if intent.order_type != OrderType.LIMIT:
            return self._skip(
                intent.order_id,
                "not_limit_order",
                broker_order_id=broker_order.broker_order_id,
            )

        route = self._router.route(intent)
        if route.tactic != ExecutionTactic.PASSIVE_LIMIT:
            return self._skip(
                intent.order_id,
                "not_passive_route",
                broker_order_id=broker_order.broker_order_id,
            )

        submitted_at = broker_order.submitted_at or intent.created_at
        age_seconds = (self._clock.now() - submitted_at).total_seconds()
        if age_seconds < self._policy.reprice_interval_seconds:
            return self._skip(
                intent.order_id,
                "not_due",
                broker_order_id=broker_order.broker_order_id,
            )

        reprice_count = await self._reprice_count(intent.order_id)
        if reprice_count >= self._policy.max_reprices_per_order:
            return self._skip(
                intent.order_id,
                "max_reprices_exceeded",
                broker_order_id=broker_order.broker_order_id,
            )

        reference_price = self._reference_price_lookup(intent)
        if reference_price is None:
            return self._skip(
                intent.order_id,
                "missing_reference_price",
                broker_order_id=broker_order.broker_order_id,
            )

        new_limit_price = self._replacement_limit_price(intent, reference_price)
        if new_limit_price is None:
            return self._skip(
                intent.order_id,
                "reprice_threshold_not_met",
                broker_order_id=broker_order.broker_order_id,
            )

        action: PassiveRepriceAction = (
            "escalated"
            if self._adverse_drift_bps(intent, reference_price)
            >= self._policy.adverse_drift_escalate_bps
            else "cancelled"
        )
        return await self._cancel_and_maybe_replace(
            intent=intent,
            broker_order_id=broker_order.broker_order_id,
            route=route,
            action=action,
            new_limit_price=new_limit_price,
        )

    async def _cancel_and_maybe_replace(
        self,
        *,
        intent: OrderIntent,
        broker_order_id: str,
        route: VenueRoute,
        action: PassiveRepriceAction,
        new_limit_price: Decimal,
    ) -> PassiveRepriceDecision:
        request = CancelReplaceRequest(
            request_id=uuid.uuid4(),
            order_id=intent.order_id,
            broker_order_id=broker_order_id,
            requested_at=self._clock.now(),
            new_limit_price=new_limit_price,
            reason=f"passive_reprice:{action}",
        )
        try:
            await self._router.cancel_replace(request)
        except Exception as exc:
            log.warning(
                "passive_reprice.cancel_failed",
                order_id=str(intent.order_id),
                broker_order_id=broker_order_id,
                error=str(exc),
            )
            return PassiveRepriceDecision(
                order_id=intent.order_id,
                action="failed",
                reason=f"cancel_failed: {exc}",
                broker_order_id=broker_order_id,
                new_limit_price=new_limit_price,
            )

        if self._replacement_factory is None:
            return PassiveRepriceDecision(
                order_id=intent.order_id,
                action=action,
                reason="cancelled_without_replacement_factory",
                broker_order_id=broker_order_id,
                new_limit_price=new_limit_price,
            )

        replacement = self._replacement_factory(
            intent,
            new_limit_price=new_limit_price,
            route=route,
            requested_at=self._clock.now(),
        )
        if replacement is None:
            return PassiveRepriceDecision(
                order_id=intent.order_id,
                action=action,
                reason="replacement_factory_skipped",
                broker_order_id=broker_order_id,
                new_limit_price=new_limit_price,
            )

        try:
            await self._broker.place_order(replacement)
        except Exception as exc:
            log.warning(
                "passive_reprice.replacement_failed",
                order_id=str(intent.order_id),
                replacement_order_id=str(replacement.order_id),
                broker_order_id=broker_order_id,
                error=str(exc),
            )
            return PassiveRepriceDecision(
                order_id=intent.order_id,
                action="failed",
                reason=f"replacement_failed: {exc}",
                broker_order_id=broker_order_id,
                replacement_order_id=replacement.order_id,
                new_limit_price=new_limit_price,
            )

        return PassiveRepriceDecision(
            order_id=intent.order_id,
            action="replaced" if action == "cancelled" else action,
            reason="replacement_submitted",
            broker_order_id=broker_order_id,
            replacement_order_id=replacement.order_id,
            new_limit_price=new_limit_price,
        )

    async def _reprice_count(self, order_id: uuid.UUID) -> int:
        if self._order_state is None:
            return 0
        events = await self._order_state.list_events(order_id)
        return sum(
            1
            for event in events
            if event.payload.get("source") == "passive_reprice"
            and event.payload.get("action") in {"cancelled", "replaced", "escalated"}
        )

    def _replacement_limit_price(
        self,
        intent: OrderIntent,
        reference_price: Decimal,
    ) -> Decimal | None:
        if intent.limit_price is None or intent.limit_price <= 0 or reference_price <= 0:
            return None
        improvement_bps = abs(reference_price - intent.limit_price) / intent.limit_price
        if float(improvement_bps * Decimal("10000")) < self._policy.min_reprice_improvement_bps:
            return None
        return max(reference_price, Decimal("0.01"))

    @staticmethod
    def _adverse_drift_bps(intent: OrderIntent, reference_price: Decimal) -> float:
        if intent.limit_price is None or intent.limit_price <= 0:
            return 0.0
        if intent.side == OrderSide.BUY:
            drift = max(reference_price - intent.limit_price, Decimal("0"))
        else:
            drift = max(intent.limit_price - reference_price, Decimal("0"))
        return float(drift / intent.limit_price * Decimal("10000"))

    @staticmethod
    def _skip(
        order_id: uuid.UUID,
        reason: str,
        *,
        broker_order_id: str | None = None,
    ) -> PassiveRepriceDecision:
        return PassiveRepriceDecision(
            order_id=order_id,
            action="skipped",
            reason=reason,
            broker_order_id=broker_order_id,
        )
