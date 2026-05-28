"""Per-intent submission flow for the submit-orders controller."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderStateEventType,
    OrderStatus,
)
from quant_platform.core.events import KillSwitchActivated, OrderSubmissionUncertain, OrderSubmitted
from quant_platform.core.exceptions import (
    BrokerAckTimeoutError,
    BrokerSubmissionError,
    BrokerUnavailableError,
)
from quant_platform.telemetry.metrics import (
    observe_order_submit_latency,
    record_order_submitted,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import (
        BrokerAck,
        CashConstraintEngine,
        EventBus,
        ExecutionPolicy,
    )
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

log = structlog.get_logger(__name__)


class SubmitOrderSubmissionMixin:
    """Per-order branch orchestration for broker submission."""

    _bus: EventBus
    _cash: CashConstraintEngine
    _engine_name: str
    _policy: ExecutionPolicy

    if TYPE_CHECKING:

        async def _pretrade_block_reason(
            self,
            intent: OrderIntent,
            account: AccountSnapshot | None,
        ) -> str | None: ...

        async def _reject_intent(
            self,
            intent: OrderIntent,
            reason: str,
            *,
            metric_reason: str,
            release_reservation_reason: str | None = None,
        ) -> None: ...

        def _check_post_approve_staleness(
            self,
            intent: OrderIntent,
            account: AccountSnapshot | None,
        ) -> str | None: ...

        async def _place_with_retry(self, intent: OrderIntent) -> BrokerAck: ...

        async def _activate_kill_switch(self, reason: str, *, activated_by: str) -> None: ...

        def _now(self) -> datetime: ...

        async def _append_order_state(
            self,
            intent: OrderIntent,
            event_type: OrderStateEventType,
            status: OrderStatus,
            *,
            broker_order_id: str | None = None,
            reason: str = "",
        ) -> None: ...

    async def _submit_one(
        self,
        intent: OrderIntent,
        account: AccountSnapshot | None,
    ) -> uuid.UUID | None:
        block_reason = await self._pretrade_block_reason(intent, account)
        if block_reason is not None:
            await self._reject_intent(
                intent,
                f"compliance: {block_reason}",
                metric_reason="compliance_block",
                release_reservation_reason=(
                    "compliance_block" if intent.cash_reservation_id is not None else None
                ),
            )
            return None

        decision = self._policy.can_submit(intent)
        if not decision.approved:
            log.warning(
                "submit_orders.blocked_by_policy",
                order_id=str(intent.order_id),
                reason=decision.reason,
            )
            await self._reject_intent(
                intent,
                f"execution policy: {decision.reason}",
                metric_reason="execution_policy",
            )
            return None

        stale_reason = self._check_post_approve_staleness(intent, account)
        if stale_reason is not None:
            log.warning(
                "submit_orders.post_approve_stale",
                order_id=str(intent.order_id),
                reason=stale_reason,
            )
            await self._reject_intent(
                intent,
                f"post_approve_staleness: {stale_reason}",
                metric_reason="post_approve_staleness",
                release_reservation_reason=(
                    "stale_approval" if intent.cash_reservation_id is not None else None
                ),
            )
            return None

        if getattr(self._policy, "kill_switch_active", False):
            reason = "kill switch activated between approval and submission"
            log.warning(
                "submit_orders.kill_switch_pre_submission",
                order_id=str(intent.order_id),
            )
            await self._reject_intent(
                intent,
                reason,
                metric_reason="kill_switch_pre_submission",
                release_reservation_reason=(
                    "kill_switch_pre_submission" if intent.cash_reservation_id is not None else None
                ),
            )
            return None

        submit_started = time.perf_counter()
        try:
            ack = await self._place_with_retry(intent)
        except BrokerAckTimeoutError as exc:
            await self._handle_ack_timeout(intent, exc, submit_started)
            return None
        except BrokerSubmissionError as exc:
            observe_order_submit_latency(
                self._engine_name, "error", time.perf_counter() - submit_started
            )
            log.error(
                "submit_orders.broker_rejected",
                order_id=str(intent.order_id),
                error=str(exc),
            )
            await self._reject_intent(
                intent,
                f"broker rejection: {exc}",
                metric_reason="broker_rejection",
            )
            return None
        except BrokerUnavailableError as exc:
            observe_order_submit_latency(
                self._engine_name, "error", time.perf_counter() - submit_started
            )
            log.error(
                "submit_orders.broker_unavailable",
                order_id=str(intent.order_id),
                error=str(exc),
            )
            await self._reject_intent(
                intent,
                f"broker unavailable: {exc}",
                metric_reason="broker_unavailable",
            )
            return None

        return await self._record_acknowledged(intent, ack, submit_started)

    async def _handle_ack_timeout(
        self,
        intent: OrderIntent,
        exc: BrokerAckTimeoutError,
        submit_started: float,
    ) -> None:
        observe_order_submit_latency(
            self._engine_name, "uncertain", time.perf_counter() - submit_started
        )
        reason = f"broker ack timeout: {exc}"
        log.error(
            "submit_orders.broker_ack_timeout_uncertain",
            order_id=str(intent.order_id),
            broker_order_id=exc.broker_order_id,
            error=str(exc),
        )
        if not getattr(self._policy, "kill_switch_active", False):
            await self._activate_kill_switch(
                reason,
                activated_by="broker_gateway",
            )
        occurred_at = self._now()
        await self._append_order_state(
            intent,
            OrderStateEventType.UNCERTAIN,
            OrderStatus.SUBMITTED,
            broker_order_id=exc.broker_order_id,
            reason=reason,
        )
        await self._bus.publish(
            OrderSubmissionUncertain(
                event_id=uuid.uuid4(),
                occurred_at=occurred_at,
                order_id=intent.order_id,
                broker_order_id=exc.broker_order_id,
                reason=reason,
            )
        )
        await self._bus.publish(
            KillSwitchActivated(
                event_id=uuid.uuid4(),
                occurred_at=occurred_at,
                activated_by="broker_gateway",
                reason=reason,
            )
        )

    async def _record_acknowledged(
        self,
        intent: OrderIntent,
        ack: BrokerAck,
        submit_started: float,
    ) -> uuid.UUID:
        observe_order_submit_latency(
            self._engine_name, "acked", time.perf_counter() - submit_started
        )

        self._policy.record_submission(intent.order_id)

        if hasattr(self._cash, "mark_order_submitted"):
            self._cash.mark_order_submitted(intent.order_id)

        await self._append_order_state(
            intent,
            OrderStateEventType.ACKNOWLEDGED,
            OrderStatus.SUBMITTED,
            broker_order_id=ack.broker_order_id,
        )
        record_order_submitted(self._engine_name)
        await self._bus.publish(
            OrderSubmitted(
                event_id=uuid.uuid4(),
                occurred_at=ack.acknowledged_at,
                order_id=intent.order_id,
                broker_order_id=ack.broker_order_id,
            )
        )
        log.info(
            "submit_orders.submitted",
            order_id=str(intent.order_id),
            broker_order_id=ack.broker_order_id,
        )
        return intent.order_id


__all__ = ["SubmitOrderSubmissionMixin"]
