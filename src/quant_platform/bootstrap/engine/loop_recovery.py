"""Kill-switch recovery checks used by the engine loop."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.bootstrap.engine.loop_types import KillSwitchRecoveryAssessment
from quant_platform.services.execution_service.reconciliation.reconciliation_discrepancies import (
    DiscrepancyResolution,
    classify_position_discrepancies,
)
from quant_platform.telemetry.metrics import set_kill_switch

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.positions import PositionSnapshot

log = structlog.get_logger(__name__)


async def refresh_kill_switch_state(session: Session) -> None:
    """Refresh in-memory execution policy from the durable kill-switch store."""

    store = getattr(session, "kill_switch_store", None)
    policy = getattr(session, "execution_policy", None)
    if store is None or policy is None:
        return
    state = await store.get()
    hydrate = getattr(policy, "hydrate_kill_switch", None)
    if callable(hydrate):
        hydrate(active=bool(state.active), reason=state.reason)
    set_kill_switch(bool(state.active))


async def assess_kill_switch_recovery(session: Session) -> KillSwitchRecoveryAssessment:
    """Run read-only recovery checks without clearing the kill switch."""

    await refresh_kill_switch_state(session)
    if not kill_switch_active(session):
        return KillSwitchRecoveryAssessment(
            active=False,
            ready_for_operator_clear=False,
            broker_connected=False,
            open_orders=None,
            operator_discrepancies=None,
            cash_drift_ok=None,
            detail="kill switch is clear",
        )

    try:
        from quant_platform.core.contracts import BrokerHealthStatus

        health = await session.account_broker.health_check()
        broker_connected = health.status == BrokerHealthStatus.CONNECTED
    except Exception as exc:
        return KillSwitchRecoveryAssessment(
            active=True,
            ready_for_operator_clear=False,
            broker_connected=False,
            open_orders=None,
            operator_discrepancies=None,
            cash_drift_ok=None,
            detail=f"broker health check failed: {exc}",
        )

    open_orders = await _open_order_count(session)
    operator_discrepancies = await _operator_discrepancy_count(session)
    cash_drift_ok = await _cash_drift_ok(session)
    ready = (
        broker_connected
        and open_orders == 0
        and operator_discrepancies == 0
        and cash_drift_ok is not False
    )
    detail = (
        "ready for operator clear"
        if ready
        else "waiting for broker, open-order, reconciliation, or cash-drift checks"
    )
    return KillSwitchRecoveryAssessment(
        active=True,
        ready_for_operator_clear=ready,
        broker_connected=broker_connected,
        open_orders=open_orders,
        operator_discrepancies=operator_discrepancies,
        cash_drift_ok=cash_drift_ok,
        detail=detail,
    )


def kill_switch_active(session: Session | None) -> bool:
    """Return whether the session execution policy is kill-switch blocked."""

    if session is None:
        return False
    policy = getattr(session, "execution_policy", None)
    if policy is None:
        return False
    active = getattr(policy, "kill_switch_active", False)
    if isinstance(active, bool):
        return active
    checker = getattr(policy, "is_kill_switch_active", None)
    return bool(checker()) if callable(checker) else bool(active)


async def _open_order_count(session: Session) -> int | None:
    fetch_open_orders = getattr(session.account_broker, "fetch_open_orders", None)
    if not callable(fetch_open_orders):
        return None
    try:
        return len(await fetch_open_orders())
    except Exception as exc:
        log.warning("engine_loop.kill_switch.open_orders_failed", error=str(exc))
        return None


async def _operator_discrepancy_count(session: Session) -> int | None:
    try:
        broker_positions: list[PositionSnapshot] = await session.account_broker.sync_positions()
        internal_snapshot = await session.position_repo.get_latest_snapshot()
        discrepancies = classify_position_discrepancies(
            broker_positions=broker_positions,
            internal_snapshot=internal_snapshot,
            detected_at=session.clock.now(),
            auto_correct_threshold=session.settings.risk.auto_correct_threshold,
        )
    except Exception as exc:
        log.warning("engine_loop.kill_switch.reconciliation_assessment_failed", error=str(exc))
        return None
    return sum(
        1
        for discrepancy in discrepancies
        if discrepancy.resolution == DiscrepancyResolution.OPERATOR_ACTION_REQUIRED
    )


async def _cash_drift_ok(session: Session) -> bool | None:
    try:
        account = await session.account_broker.sync_account()
        settled_cash = getattr(session.cash_engine, "settled_cash", None)
        if settled_cash is None or account.source != "broker":
            return None
        drift = Decimal(str(settled_cash)) - Decimal(str(account.settled_cash))
        return abs(drift) <= session.settings.cash.drift_tolerance_usd
    except Exception as exc:
        log.warning("engine_loop.kill_switch.cash_drift_assessment_failed", error=str(exc))
        return None


__all__ = [
    "assess_kill_switch_recovery",
    "kill_switch_active",
    "refresh_kill_switch_state",
]
