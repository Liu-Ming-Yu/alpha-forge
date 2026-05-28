"""Strategy-cycle runtime orchestration helpers."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.application.runtime.state import CycleResult, Session
from quant_platform.core.events import KillSwitchActivated
from quant_platform.engines.session.cycle_guards import (
    activate_and_publish_kill_switch,
    durable_kill_switch,
    publish_execution_state_metrics,
    run_pre_cycle_housekeeping,
    sync_account_or_halt,
)
from quant_platform.engines.session.cycle_lifecycle import (
    record_nav_snapshot,
    strategy_cycle_lock,
)
from quant_platform.engines.session.cycle_steps import (
    apply_cycle_passive_reprice,
    approve_cycle_orders,
    build_cycle_portfolio_target,
    detect_cycle_regime,
    drain_cycle_lifecycle_events,
    generate_cycle_signals,
    plan_cycle_orders,
    publish_cycle_regime,
    refresh_cycle_vol_forecasts,
    submit_cycle_orders,
)
from quant_platform.engines.session.regime_stats import compute_market_stats_from_store
from quant_platform.engines.session.runtime import hydrate_session_state
from quant_platform.telemetry.metrics import set_kill_switch

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.domain.research import StrategyRun
    from quant_platform.core.domain.signals import RegimeState
    from quant_platform.engines.session.strategy_cycle_types import MarketStatsReader

log = structlog.get_logger(__name__)

CycleRunner = Callable[..., Awaitable[CycleResult]]
SessionHydrator = Callable[[Session], Awaitable[None]]
NavSnapshotRecorder = Callable[[Session, uuid.UUID], Awaitable[None]]
StrategyCycleLock = Callable[[Session, uuid.UUID], AbstractAsyncContextManager[object]]

__all__ = [
    "durable_kill_switch",
    "record_nav_snapshot",
    "run_strategy_cycle",
    "run_strategy_cycle_unlocked",
    "strategy_cycle_lock",
]


class KillSwitcher(Protocol):
    def __call__(self, policy: object, reason: str, *, activated_by: str) -> Awaitable[None]: ...


async def run_strategy_cycle_unlocked(
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
    strategy_run: StrategyRun,
    market_prices: dict[uuid.UUID, Decimal] | None = None,
    regime: RegimeState | None = None,
    as_of: datetime | None = None,
    lock: object | None = None,
    market_stats_reader: MarketStatsReader = compute_market_stats_from_store,
) -> CycleResult:
    """Execute one complete rebalance cycle without acquiring the outer lock."""
    _prices = market_prices or {}
    _as_of = as_of or session.clock.now()
    if (
        session.signal_ctrl is None
        or session.portfolio_ctrl is None
        or session.order_planner is None
        or session.regime_detector is None
    ):
        raise RuntimeError("strategy cycle session is missing required controllers")

    halt_result = await run_pre_cycle_housekeeping(session, strategy_run)
    if halt_result is not None:
        return halt_result

    account, halt_result = await sync_account_or_halt(session)
    if halt_result is not None:
        return halt_result

    engine_name = strategy_run.strategy_name
    await apply_cycle_passive_reprice(
        session=session,
        market_prices=_prices,
        engine_name=engine_name,
    )

    signals = await generate_cycle_signals(
        session=session,
        feature_data=feature_data,
        strategy_run=strategy_run,
        as_of=_as_of,
        engine_name=engine_name,
    )

    _regime, halt_result = await detect_cycle_regime(
        session=session,
        supplied_regime=regime,
        as_of=_as_of,
        engine_name=engine_name,
        market_stats_reader=market_stats_reader,
    )
    if halt_result is not None:
        return halt_result
    if _regime is None:
        raise RuntimeError("regime detection returned no regime without a halt result")

    await publish_cycle_regime(session, _regime)
    refresh_cycle_vol_forecasts(session, feature_data)

    target = await build_cycle_portfolio_target(
        session=session,
        signals=signals,
        regime=_regime,
        account=account,
        engine_name=engine_name,
    )
    if target is None:
        log.warning("strategy_cycle.target_rejected_by_risk_policy", as_of=str(_as_of))
        return CycleResult(
            signals=signals,
            target=None,
            approved=[],
            rejected=[],
            submitted_ids=[],
            fills=[],
        )

    intents = plan_cycle_orders(
        session=session,
        target=target,
        account=account,
        market_prices=_prices,
        strategy_run=strategy_run,
        engine_name=engine_name,
    )

    if not intents:
        log.info("strategy_cycle.no_orders_planned", target_id=str(target.target_id))
        return CycleResult(
            signals=signals,
            target=target,
            approved=[],
            rejected=[],
            submitted_ids=[],
            fills=[],
        )

    approved, rejected = await approve_cycle_orders(
        session=session,
        intents=intents,
        account=account,
        engine_name=engine_name,
    )

    if lock is not None and getattr(lock, "lease_lost", False):
        lease_reason = (
            "distributed lock lease lost before order submission; "
            "aborting order submission to avoid concurrent state mutation"
        )
        log.error("strategy_cycle.lease_lost_pre_submit", reason=lease_reason)
        await activate_and_publish_kill_switch(
            session,
            reason=lease_reason,
            activated_by="distributed_lock",
        )
        raise RuntimeError(lease_reason)
    submitted_ids = await submit_cycle_orders(
        session=session,
        approved=approved,
        account=account,
        engine_name=engine_name,
    )

    fills = await drain_cycle_lifecycle_events(
        session=session,
        engine_name=engine_name,
        wait_timeout_seconds=session.settings.execution.post_submit_lifecycle_drain_seconds,
        poll_interval_seconds=session.settings.execution.lifecycle_drain_poll_seconds,
    )

    publish_execution_state_metrics(session)

    log.info(
        "strategy_cycle.complete",
        signals=len(signals),
        orders_planned=len(intents),
        approved=len(approved),
        rejected=len(rejected),
        submitted=len(submitted_ids),
        fills=len(fills),
        as_of=str(_as_of),
    )

    return CycleResult(
        signals=signals,
        target=target,
        approved=approved,
        rejected=rejected,
        submitted_ids=submitted_ids,
        fills=fills,
    )


async def run_strategy_cycle(
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
    strategy_run: StrategyRun,
    market_prices: dict[uuid.UUID, Decimal] | None = None,
    regime: RegimeState | None = None,
    as_of: datetime | None = None,
    *,
    cycle_runner: CycleRunner = run_strategy_cycle_unlocked,
    hydrator: SessionHydrator = hydrate_session_state,
    kill_switcher: KillSwitcher = durable_kill_switch,
    nav_snapshot_recorder: NavSnapshotRecorder = record_nav_snapshot,
    lock_context: StrategyCycleLock = strategy_cycle_lock,
) -> CycleResult:
    """Execute one complete rebalance cycle with optional distributed locking."""
    await hydrator(session)
    cycle_started_at = session.clock.now()
    async with lock_context(session, strategy_run.run_id) as lock:
        try:
            result = await cycle_runner(
                session=session,
                feature_data=feature_data,
                strategy_run=strategy_run,
                market_prices=market_prices,
                regime=regime,
                as_of=as_of,
                lock=lock,
            )
        except Exception as _cycle_exc:
            _cycle_reason = (
                f"unhandled exception in strategy cycle: {type(_cycle_exc).__name__}: {_cycle_exc}"
            )
            log.error(
                "strategy_cycle.unhandled_exception",
                reason=_cycle_reason,
                exc_info=True,
            )
            await kill_switcher(session.execution_policy, _cycle_reason, activated_by="cycle_guard")
            set_kill_switch(True)
            await session.event_bus.publish(
                KillSwitchActivated(
                    event_id=uuid.uuid4(),
                    occurred_at=session.clock.now(),
                    activated_by="cycle_guard",
                    reason=_cycle_reason,
                )
            )
            raise
    if getattr(lock, "lease_lost", False):
        reason = (
            "strategy cycle lock lease was lost before cycle completion; "
            "execution halted to avoid concurrent state mutation"
        )
        await kill_switcher(session.execution_policy, reason, activated_by="distributed_lock")
        await session.event_bus.publish(
            KillSwitchActivated(
                event_id=uuid.uuid4(),
                occurred_at=session.clock.now(),
                activated_by="distributed_lock",
                reason=reason,
            )
        )
        log.error("strategy_cycle.lock_lease_lost", strategy_run_id=str(strategy_run.run_id))
        raise RuntimeError(reason)
    if session.settings.storage.redis_url:
        elapsed = (session.clock.now() - cycle_started_at).total_seconds()
        ttl = float(session.settings.storage.distributed_lock_ttl_seconds)
        if ttl > 0 and elapsed >= ttl * 0.80:
            log.warning(
                "strategy_cycle.lock_ttl_margin_low",
                strategy_run_id=str(strategy_run.run_id),
                elapsed_seconds=elapsed,
                ttl_seconds=ttl,
            )
    await nav_snapshot_recorder(session, strategy_run.run_id)
    return result
