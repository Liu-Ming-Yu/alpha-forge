"""Broker-state reconciliation controller."""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.events import (
    CashDriftDetected,
    KillSwitchActivated,
    ReconciliationCompleted,
)
from quant_platform.telemetry.metrics import (
    record_reconciliation_mismatch,
    set_cash_drift,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import EventBus, ExecutionPolicy
    from quant_platform.services.execution_service.reconciliation import ReconciliationEngine

log = structlog.get_logger(__name__)


class ReconcileBrokerStateControllerImpl:
    """Run a full reconciliation cycle and publish the summary event.

    Args:
        engine: ReconciliationEngine that performs the actual diff.
        event_bus: EventBus for publishing the summary event.
        execution_policy: ExecutionPolicy to activate kill switch if needed.
    """

    def __init__(
        self,
        engine: ReconciliationEngine,
        event_bus: EventBus,
        execution_policy: ExecutionPolicy,
        engine_name: str = "default",
    ) -> None:
        self._engine = engine
        self._bus = event_bus
        self._policy = execution_policy
        self._engine_name = engine_name

    async def reconcile(self, strategy_run_id: uuid.UUID) -> None:
        """Run reconciliation and publish the result."""
        summary: ReconciliationCompleted = await self._engine.reconcile(strategy_run_id)

        await self._bus.publish(summary)

        drift_event: CashDriftDetected | None = getattr(self._engine, "last_cash_drift_event", None)
        if drift_event is not None:
            await self._bus.publish(drift_event)

        for mismatch_type, count in getattr(summary, "discrepancy_counts", {}).items():
            for _ in range(int(count)):
                record_reconciliation_mismatch(self._engine_name, str(mismatch_type))
        cash_drift = summary.cash_drift_usd
        if cash_drift is not None:
            with contextlib.suppress(TypeError, ValueError):
                set_cash_drift(self._engine_name, float(cash_drift))

        if summary.requires_operator_action:
            reason = (
                f"reconciliation found {summary.discrepancies_found} discrepancies "
                f"requiring operator action"
            )
            activate_durable = getattr(self._policy, "activate_kill_switch_durable", None)
            if activate_durable is not None:
                await activate_durable(
                    reason,
                    activated_by="reconciliation",
                )
            elif hasattr(self._policy, "activate_kill_switch"):
                activate = self._policy.activate_kill_switch
                activate(reason, activated_by="reconciliation")
            await self._bus.publish(
                KillSwitchActivated(
                    event_id=uuid.uuid4(),
                    occurred_at=summary.occurred_at,
                    activated_by="reconciliation",
                    reason="unresolvable position discrepancy detected",
                )
            )
            log.warning(
                "reconcile.kill_switch_activated",
                strategy_run_id=str(strategy_run_id),
                discrepancies=summary.discrepancies_found,
            )
