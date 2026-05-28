"""Support behaviors for the submit-orders controller."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
)
from quant_platform.core.events import ComplianceViolationWarning, OrderRejected
from quant_platform.core.exceptions import BrokerAckTimeoutError, BrokerUnavailableError
from quant_platform.telemetry.metrics import record_order_rejected

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import (
        BrokerAck,
        BrokerOrderRoutingGateway,
        CashConstraintEngine,
        Clock,
        EventBus,
        ExecutionPolicy,
        OrderRepository,
        OrderStateStore,
        RiskPolicy,
    )
    from quant_platform.core.domain.portfolio import RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.execution_service.orders.pretrade_compliance import (
        PreTradeComplianceChecker,
    )
    from quant_platform.services.execution_service.support.circuit_breaker import CircuitBreaker

log = structlog.get_logger(__name__)


class SubmitOrderSupportMixin:
    """Retry, rejection, and state-recording helpers for submit orchestration."""

    _broker: BrokerOrderRoutingGateway
    _bus: EventBus
    _cash: CashConstraintEngine
    _circuit_breaker: CircuitBreaker | None
    _clock: Clock
    _compliance: PreTradeComplianceChecker | None
    _engine_name: str
    _limits: RiskLimits | None
    _order_state: OrderStateStore | None
    _orders: OrderRepository | None
    _policy: ExecutionPolicy
    _retry_attempts: int
    _retry_base_delay: float
    _risk: RiskPolicy | None

    async def _reject_intent(
        self,
        intent: OrderIntent,
        reason: str,
        *,
        metric_reason: str,
        release_reservation_reason: str | None = None,
    ) -> None:
        if release_reservation_reason is not None and intent.cash_reservation_id is not None:
            self._cash.release_reservation(
                intent.cash_reservation_id,
                reason=release_reservation_reason,
            )
        if hasattr(self._cash, "cancel_order"):
            self._cash.cancel_order(intent.order_id, reason)
        await self._mark_terminal(intent.order_id, reason)
        await self._append_order_state(
            intent,
            OrderStateEventType.REJECTED,
            OrderStatus.REJECTED,
            reason=reason,
        )
        record_order_rejected(self._engine_name, metric_reason)
        await self._bus.publish(
            OrderRejected(
                event_id=uuid.uuid4(),
                occurred_at=intent.created_at,
                order_id=intent.order_id,
                reason=reason,
            )
        )

    async def _place_with_retry(self, intent: OrderIntent) -> BrokerAck:
        """Call broker.place_order with exponential backoff on unavailability."""
        last_exc: BrokerUnavailableError | None = None
        for attempt in range(self._retry_attempts):
            try:
                if self._circuit_breaker is not None:
                    return await self._circuit_breaker.call(
                        lambda: self._broker.place_order(intent)
                    )
                return await self._broker.place_order(intent)
            except BrokerAckTimeoutError:
                raise
            except BrokerUnavailableError as exc:
                last_exc = exc
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_base_delay * (2**attempt)
                    log.warning(
                        "submit_orders.broker_unavailable_retrying",
                        order_id=str(intent.order_id),
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
        if last_exc is None:
            raise BrokerUnavailableError("broker unavailable retry exhausted without an exception")
        raise last_exc

    async def _pretrade_block_reason(
        self,
        intent: OrderIntent,
        account: AccountSnapshot | None,
    ) -> str | None:
        """Run submit-time compliance and return a block reason when needed."""
        if self._compliance is None or account is None:
            return None

        pos_map = {position.instrument_id: position for position in account.positions}
        violations = self._compliance.check(intent, account, pos_map)
        block_violations = [violation for violation in violations if violation.severity == "BLOCK"]
        warn_violations = [violation for violation in violations if violation.severity == "WARN"]

        for warning in warn_violations:
            log.warning(
                "pretrade_compliance.warn",
                order_id=str(intent.order_id),
                rule=warning.rule,
                detail=warning.detail,
            )
            await self._bus.publish(
                ComplianceViolationWarning(
                    event_id=uuid.uuid4(),
                    occurred_at=self._now(),
                    order_id=intent.order_id,
                    rule=warning.rule,
                    detail=warning.detail,
                )
            )

        if not block_violations:
            return None

        block_reason = "; ".join(violation.detail for violation in block_violations)
        log.error(
            "pretrade_compliance.blocked",
            order_id=str(intent.order_id),
            rules=[violation.rule for violation in block_violations],
            reason=block_reason,
        )
        return block_reason

    def _check_post_approve_staleness(
        self,
        intent: OrderIntent,
        account: AccountSnapshot | None,
    ) -> str | None:
        """Return a rejection reason when submit-time state invalidates approval."""
        if self._risk is not None and self._limits is not None and account is not None:
            risk_decision = self._risk.check_order_limits(intent, account, self._limits)
            if not risk_decision.approved:
                return f"risk revalidation failed: {risk_decision.reason}"

        if (
            intent.side == OrderSide.BUY
            and intent.cash_reservation_id is not None
            and hasattr(self._cash, "is_reservation_active")
        ):
            is_reservation_active = getattr(self._cash, "is_reservation_active", None)
            if is_reservation_active is None:
                return None
            if not bool(is_reservation_active(intent.cash_reservation_id)):
                return "cash reservation no longer active"

        return None

    async def _mark_terminal(self, order_id: uuid.UUID, reason: str) -> None:
        if self._orders is None:
            return
        mark_terminal = getattr(self._orders, "mark_terminal", None)
        if mark_terminal is not None:
            await mark_terminal(order_id, reason)

    async def _append_order_state(
        self,
        intent: OrderIntent,
        event_type: OrderStateEventType,
        status: OrderStatus,
        *,
        broker_order_id: str | None = None,
        reason: str = "",
    ) -> None:
        if self._order_state is None:
            return
        await self._order_state.append(
            OrderStateEvent(
                event_id=uuid.uuid4(),
                order_id=intent.order_id,
                event_type=event_type,
                occurred_at=self._now(),
                status=status,
                broker_order_id=broker_order_id,
                idempotency_key=f"{intent.order_id}:{event_type.value}:{broker_order_id or ''}",
                payload={"reason": reason} if reason else {},
            )
        )

    def _now(self) -> datetime:
        return self._clock.now()

    async def _activate_kill_switch(self, reason: str, *, activated_by: str) -> None:
        activate_durable = getattr(self._policy, "activate_kill_switch_durable", None)
        if activate_durable is not None:
            await activate_durable(
                reason,
                activated_by=activated_by,
            )
        elif hasattr(self._policy, "activate_kill_switch"):
            activate = self._policy.activate_kill_switch
            activate(reason, activated_by=activated_by)
