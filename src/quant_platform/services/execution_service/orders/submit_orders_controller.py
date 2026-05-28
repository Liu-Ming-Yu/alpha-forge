"""Submit-order execution controller implementation.

SubmitOrdersControllerImpl:
    Checks ExecutionPolicy, submits via BrokerGateway, notifies CashLedger,
    and emits OrderSubmitted events.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.execution_service.orders.submit_order_submission import (
    SubmitOrderSubmissionMixin,
)
from quant_platform.services.execution_service.orders.submit_order_support import (
    SubmitOrderSupportMixin,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts import (
        BrokerOrderRoutingGateway,
        CashConstraintEngine,
        Clock,
        EventBus,
        ExecutionPolicy,
        OrderRepository,
        OrderStateStore,
        RiskPolicy,
    )
    from quant_platform.core.domain.orders import (
        OrderIntent,
    )
    from quant_platform.core.domain.portfolio import RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.execution_service.orders.pretrade_compliance import (
        PreTradeComplianceChecker,
    )
    from quant_platform.services.execution_service.support.circuit_breaker import CircuitBreaker

log = structlog.get_logger(__name__)

__all__ = ["SubmitOrdersControllerImpl"]


class _UtcClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def today(self) -> date:
        return self.now().date()


class SubmitOrdersControllerImpl(SubmitOrderSubmissionMixin, SubmitOrderSupportMixin):
    """Submit approved orders to the broker, enforcing execution policy.

    Pipeline per intent:
        1. Re-validate the execution policy (kill switch + throttle).
        2. Re-validate risk.check_order_limits against the submit-time
           ``AccountSnapshot``; this is the approve-vs-submit TOCTOU guard
           introduced in Phase 1.4 of the parity plan.  Any drift that
           would have caused approve to fail now causes submit to reject
           cleanly, releasing the reservation.
        3. Re-validate the cash reservation is still ACTIVE (buy side only).
        4. Submit to the broker, handling rejection/unavailability.

    Args:
        broker: BrokerOrderRoutingGateway for placing/cancelling orders.
        execution_policy: ExecutionPolicy (throttle + kill switch).
        cash_engine: CashConstraintEngine to mark submitted orders and
            release reservations on failure.
        event_bus: EventBus for emitting submission events.
        risk_policy: Optional RiskPolicy for submit-time per-order
            revalidation. When ``None``, only the execution-policy gate
            applies for focused controller tests.
        limits: Active risk limits; required whenever ``risk_policy`` is
            provided.
    """

    def __init__(
        self,
        broker: BrokerOrderRoutingGateway,
        execution_policy: ExecutionPolicy,
        cash_engine: CashConstraintEngine,
        event_bus: EventBus,
        order_repo: OrderRepository | None = None,
        engine_name: str = "default",
        risk_policy: RiskPolicy | None = None,
        limits: RiskLimits | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        order_state_store: OrderStateStore | None = None,
        clock: Clock | None = None,
        compliance_checker: PreTradeComplianceChecker | None = None,
    ) -> None:
        self._broker = broker
        self._policy = execution_policy
        self._cash = cash_engine
        self._bus = event_bus
        self._orders = order_repo
        self._engine_name = engine_name
        self._risk = risk_policy
        self._limits = limits
        self._circuit_breaker = circuit_breaker
        self._order_state = order_state_store
        self._clock = clock or _UtcClock()
        self._compliance = compliance_checker
        if risk_policy is not None and limits is None:
            raise ValueError(
                "SubmitOrdersControllerImpl: limits must be provided "
                "when risk_policy is set; otherwise submit-time "
                "revalidation has nothing to validate against."
            )
        self._retry_attempts = 3
        self._retry_base_delay = 2.0

    async def submit(
        self,
        approved_intents: list[OrderIntent],
        account: AccountSnapshot | None = None,
    ) -> list[uuid.UUID]:
        """Submit approved orders, returning IDs of successfully submitted ones.

        Args:
            approved_intents: Orders previously approved by the
                ``ApproveOrdersController``.  Buy intents must carry a
                ``cash_reservation_id``; sell intents do not reserve cash.
            account: Submit-time account snapshot used for risk
                revalidation. When ``None``, submit-time risk checks are
                skipped.
        """
        submitted_ids: list[uuid.UUID] = []

        if not self._broker.capabilities.supports_order_routing:
            reason = f"broker '{self._broker.capabilities.provider}' does not support order routing"
            for intent in approved_intents:
                await self._reject_intent(
                    intent,
                    reason,
                    metric_reason="broker_no_routing",
                )
            return submitted_ids

        for intent in approved_intents:
            submitted_id = await self._submit_one(intent, account)
            if submitted_id is not None:
                submitted_ids.append(submitted_id)

        return submitted_ids
