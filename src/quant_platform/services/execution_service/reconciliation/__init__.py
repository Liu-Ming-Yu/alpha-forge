"""Broker state reconciliation orchestration.

Broker state is authoritative. The engine fetches broker and internal snapshots,
classifies differences, writes the broker-authoritative snapshot, records one
audit entry per discrepancy, and returns exactly one summary event per cycle.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.core.events import CashDriftDetected, ReconciliationCompleted
from quant_platform.services.execution_service.reconciliation.reconciliation_audit import (
    audit_cash_drift,
    audit_discrepancies,
)
from quant_platform.services.execution_service.reconciliation.reconciliation_discrepancies import (
    Discrepancy,
    DiscrepancyResolution,
    DiscrepancyType,
    classify_position_discrepancies,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import (
        AuditSink,
        BrokerSessionGateway,
        Clock,
        PositionRepository,
    )
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

log = structlog.get_logger(__name__)

__all__ = [
    "Discrepancy",
    "DiscrepancyResolution",
    "DiscrepancyType",
    "ReconciliationEngine",
]


class _CashLedgerProtocol(Protocol):
    """Structural subset of CashLedger needed by ReconciliationEngine."""

    @property
    def settled_cash(self) -> Decimal: ...

    def reset_from_snapshot(self, snapshot: AccountSnapshot) -> None: ...


class ReconciliationEngine:
    """Reconcile internal position and order state against broker state."""

    def __init__(
        self,
        broker: BrokerSessionGateway,
        position_repo: PositionRepository,
        audit_sink: AuditSink,
        clock: Clock,
        auto_correct_threshold: int = 1,
        ledger: _CashLedgerProtocol | None = None,
        cash_drift_threshold: Decimal = Decimal("1.00"),
    ) -> None:
        self._broker = broker
        self._positions = position_repo
        self._audit = audit_sink
        self._clock = clock
        self._auto_correct_threshold = auto_correct_threshold
        self._ledger = ledger
        self._cash_drift_threshold = cash_drift_threshold
        self.last_cash_drift_event: CashDriftDetected | None = None

    async def reconcile(self, strategy_run_id: uuid.UUID) -> ReconciliationCompleted:
        """Run one full reconciliation cycle."""

        now = self._clock.now()
        log.info("reconciliation.start", strategy_run_id=str(strategy_run_id))

        broker_account = await self._broker.sync_account()
        broker_positions = await self._broker.sync_positions()
        internal_snapshot = await self._positions.get_latest_snapshot()
        is_first = internal_snapshot is None

        discrepancies = self._diff_positions(
            broker_positions=broker_positions,
            internal_snapshot=internal_snapshot,
            detected_at=now,
        )
        auto_resolved = [
            d for d in discrepancies if d.resolution == DiscrepancyResolution.AUTO_CORRECTED
        ]
        operator_required = [
            d
            for d in discrepancies
            if d.resolution == DiscrepancyResolution.OPERATOR_ACTION_REQUIRED
        ]

        # A snapshot-save failure means the next reconcile cycle would
        # diff against stale internal state, which silently masks real
        # broker drift. Re-raise so the supervisor can retry the cycle
        # rather than report success on a half-applied write.
        try:
            await self._positions.save_snapshot(broker_account)
        except Exception:
            log.exception(
                "reconciliation.snapshot_save_failed",
                strategy_run_id=str(strategy_run_id),
            )
            raise

        cash_drift_usd, cash_drift_event = self._sync_cash_ledger(
            broker_account=broker_account,
            strategy_run_id=str(strategy_run_id),
            occurred_at=now,
        )

        await audit_discrepancies(
            self._audit,
            discrepancies=discrepancies,
            strategy_run_id=strategy_run_id,
            occurred_at=now,
        )
        await audit_cash_drift(
            self._audit,
            cash_drift_event=cash_drift_event,
            strategy_run_id=strategy_run_id,
        )

        summary = ReconciliationCompleted(
            event_id=uuid.uuid4(),
            occurred_at=now,
            strategy_run_id=strategy_run_id,
            discrepancies_found=len(discrepancies),
            discrepancies_resolved=len(auto_resolved),
            requires_operator_action=len(operator_required) > 0,
            is_first_reconciliation=is_first,
            cash_drift_usd=cash_drift_usd,
        )
        self._log_completion(
            strategy_run_id=strategy_run_id,
            is_first=is_first,
            broker_positions_count=len(broker_positions),
            discrepancies_count=len(discrepancies),
            auto_resolved_count=len(auto_resolved),
            operator_required_count=len(operator_required),
        )
        return summary

    def _sync_cash_ledger(
        self,
        *,
        broker_account: AccountSnapshot,
        strategy_run_id: str,
        occurred_at: datetime,
    ) -> tuple[Decimal | None, CashDriftDetected | None]:
        """Synchronize optional cash ledger and emit drift evidence if needed."""

        self.last_cash_drift_event = None
        if self._ledger is None:
            return None, None

        drift = self._ledger.settled_cash - broker_account.settled_cash
        cash_drift_event: CashDriftDetected | None = None
        if abs(drift) > self._cash_drift_threshold:
            log.critical(
                "reconciliation.cash_drift_detected",
                ledger_settled=str(self._ledger.settled_cash),
                broker_settled=str(broker_account.settled_cash),
                drift=str(drift),
                threshold=str(self._cash_drift_threshold),
            )
            cash_drift_event = CashDriftDetected(
                event_id=uuid.uuid4(),
                occurred_at=occurred_at,
                strategy_run_id=uuid.UUID(strategy_run_id),
                ledger_settled_cash=self._ledger.settled_cash,
                broker_settled_cash=broker_account.settled_cash,
                drift_usd=drift,
            )
            self.last_cash_drift_event = cash_drift_event

        self._ledger.reset_from_snapshot(broker_account)
        return drift, cash_drift_event

    def _log_completion(
        self,
        *,
        strategy_run_id: uuid.UUID,
        is_first: bool,
        broker_positions_count: int,
        discrepancies_count: int,
        auto_resolved_count: int,
        operator_required_count: int,
    ) -> None:
        if is_first:
            log.info(
                "reconciliation.complete.first_session",
                strategy_run_id=str(strategy_run_id),
                detail=(
                    "no prior internal snapshot; all broker positions adopted as initial state"
                ),
                broker_positions=broker_positions_count,
            )
            return

        log.info(
            "reconciliation.complete",
            strategy_run_id=str(strategy_run_id),
            discrepancies_found=discrepancies_count,
            auto_resolved=auto_resolved_count,
            operator_required=operator_required_count,
        )

    def _diff_positions(
        self,
        broker_positions: list[PositionSnapshot],
        internal_snapshot: AccountSnapshot | None,
        detected_at: datetime,
    ) -> list[Discrepancy]:
        """Compare broker and internal positions; classify each discrepancy."""

        return classify_position_discrepancies(
            broker_positions=broker_positions,
            internal_snapshot=internal_snapshot,
            detected_at=detected_at,
            auto_correct_threshold=self._auto_correct_threshold,
        )
