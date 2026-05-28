"""Audit helpers for broker reconciliation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.events import CashDriftDetected, ReconciliationCompleted
from quant_platform.services.execution_service.reconciliation.reconciliation_discrepancies import (
    Discrepancy,
    DiscrepancyResolution,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import AuditSink

log = structlog.get_logger(__name__)


async def audit_discrepancies(
    audit_sink: AuditSink,
    *,
    discrepancies: list[Discrepancy],
    strategy_run_id: uuid.UUID,
    occurred_at: datetime,
) -> None:
    """Record one audit event for each reconciliation discrepancy."""
    for disc in discrepancies:
        try:
            await audit_sink.record(
                event=ReconciliationCompleted(
                    event_id=uuid.uuid4(),
                    occurred_at=occurred_at,
                    strategy_run_id=strategy_run_id,
                    discrepancies_found=1,
                    discrepancies_resolved=(
                        1 if disc.resolution == DiscrepancyResolution.AUTO_CORRECTED else 0
                    ),
                    requires_operator_action=(
                        disc.resolution == DiscrepancyResolution.OPERATOR_ACTION_REQUIRED
                    ),
                ),
                context={
                    "discrepancy_id": str(disc.discrepancy_id),
                    "instrument_id": str(disc.instrument_id) if disc.instrument_id else None,
                    "type": disc.discrepancy_type,
                    "resolution": disc.resolution,
                    "internal_value": disc.internal_value,
                    "broker_value": disc.broker_value,
                    "notes": disc.notes,
                },
            )
        except Exception:
            log.exception(
                "reconciliation.audit_record_failed",
                discrepancy_id=str(disc.discrepancy_id),
            )


async def audit_cash_drift(
    audit_sink: AuditSink,
    *,
    cash_drift_event: CashDriftDetected | None,
    strategy_run_id: uuid.UUID,
) -> None:
    """Record cash-drift evidence when a reconciliation cycle detects drift."""
    if cash_drift_event is None:
        return
    try:
        await audit_sink.record(
            event=cash_drift_event,
            context={
                "drift_usd": str(cash_drift_event.drift_usd),
                "ledger_settled": str(cash_drift_event.ledger_settled_cash),
                "broker_settled": str(cash_drift_event.broker_settled_cash),
            },
        )
    except Exception:
        log.exception(
            "reconciliation.cash_drift_audit_failed",
            strategy_run_id=str(strategy_run_id),
        )
