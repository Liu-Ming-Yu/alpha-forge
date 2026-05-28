"""Pre-cycle guardrails and housekeeping for strategy-cycle orchestration."""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.runtime.state import CycleResult, Session
from quant_platform.core.events import KillSwitchActivated
from quant_platform.telemetry.metrics import set_kill_switch, set_throttle_tokens

if TYPE_CHECKING:
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research import StrategyRun

log = structlog.get_logger(__name__)


def empty_cycle_result() -> CycleResult:
    """Return the standard halted/no-op cycle result."""
    return CycleResult(
        signals=[],
        target=None,
        approved=[],
        rejected=[],
        submitted_ids=[],
        fills=[],
    )


async def durable_kill_switch(policy: object, reason: str, activated_by: str) -> None:
    """Activate the kill switch, preferring the durable store-first form."""
    if hasattr(policy, "activate_kill_switch_durable"):
        await policy.activate_kill_switch_durable(reason, activated_by=activated_by)
    elif hasattr(policy, "activate_kill_switch"):
        policy.activate_kill_switch(reason, activated_by=activated_by)


async def activate_and_publish_kill_switch(
    session: Session,
    *,
    reason: str,
    activated_by: str,
) -> None:
    """Activate execution policy and publish a KillSwitchActivated event."""
    await durable_kill_switch(session.execution_policy, reason, activated_by=activated_by)
    set_kill_switch(True)
    await session.event_bus.publish(
        KillSwitchActivated(
            event_id=uuid.uuid4(),
            occurred_at=session.clock.now(),
            activated_by=activated_by,
            reason=reason,
        )
    )


async def run_pre_cycle_housekeeping(
    session: Session,
    strategy_run: StrategyRun,
) -> CycleResult | None:
    """Run broker recovery, reservation expiry, settlement, and event drain."""
    if session.supervisor is not None:
        ready = await session.supervisor.startup_recovery(strategy_run.run_id)
        if not ready:
            log.error("strategy_cycle.startup_recovery_failed")
            return empty_cycle_result()
        healthy = await session.supervisor.poll_health(strategy_run.run_id)
        if not healthy:
            log.error("strategy_cycle.unhealthy_broker")
            return empty_cycle_result()
        await session.supervisor.cleanup_stale_day_orders(strategy_run.run_id)

    expire_stale = getattr(session.cash_engine, "expire_stale_reservations", None)
    if callable(expire_stale):
        expire_stale()

    settle_pending_buys = getattr(session.cash_engine, "settle_pending_buys", None)
    if callable(settle_pending_buys):
        settle_pending_buys(session.clock.today())
    await session.coordinator.advance_settlements()

    if session.lifecycle_feed is not None:
        pre_events = await session.lifecycle_feed.drain_lifecycle_events()
        if pre_events:
            await session.coordinator.process_lifecycle_events(pre_events)
    return None


async def sync_account_or_halt(session: Session) -> tuple[AccountSnapshot, CycleResult | None]:
    """Sync broker account and enforce cash-drift/drawdown fail-closed guards."""
    account = await session.account_broker.sync_account()
    if account.source != "broker":
        return account, None

    drift_ok, drift = session.coordinator.check_cash_drift(
        broker_settled=account.settled_cash,
        tolerance=session.settings.cash.drift_tolerance_usd,
    )
    if not drift_ok:
        session.coordinator.resync_from_broker_snapshot(account)
        await session.coordinator.purge_durable_state()
        reason = (
            "cash drift exceeded tolerance: "
            f"drift={drift}, tolerance={session.settings.cash.drift_tolerance_usd}"
        )
        await activate_and_publish_kill_switch(
            session,
            reason=reason,
            activated_by="cash_drift_guard",
        )
        log.error("strategy_cycle.cash_drift_halt", reason=reason)
        return account, empty_cycle_result()

    if session.drawdown_guard is None:
        return account, None

    dd_ok, drawdown = session.drawdown_guard.update_and_check(account.net_asset_value)
    if dd_ok:
        return account, None

    reason = (
        f"session_drawdown_halt: drawdown={float(drawdown):.4f} "
        f"exceeds limit={float(session.risk_limits.max_drawdown_halt):.4f}"
    )
    await activate_and_publish_kill_switch(
        session,
        reason=reason,
        activated_by="drawdown_guard",
    )
    log.error("strategy_cycle.session_drawdown_halt", reason=reason, drawdown=float(drawdown))
    return account, empty_cycle_result()


def publish_execution_state_metrics(session: Session) -> None:
    """Best-effort throttle and kill-switch metric surfacing."""
    tokens_fn = getattr(session.execution_policy, "available_tokens", None)
    if callable(tokens_fn):
        with contextlib.suppress(Exception):
            set_throttle_tokens(float(tokens_fn()))
    kill_fn = getattr(session.execution_policy, "is_kill_switch_active", None)
    if callable(kill_fn):
        with contextlib.suppress(Exception):
            set_kill_switch(bool(kill_fn()))
