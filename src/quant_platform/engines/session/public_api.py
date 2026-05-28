"""Session runtime API: strategy-cycle execution and durable-state hydration.

The wired public runtime operations -- ``run_strategy_cycle``,
``hydrate_session_state``, ``model_registry_preflight`` -- and the cycle hooks
they compose. These *drive* a ``Session``, so they live in ``engines`` (the run
loop). ``bootstrap/session/public_api.py`` is the symmetric *composition* API
(``create_*_session``); ``quant_platform.session`` re-exports both.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from quant_platform.engines.session.regime_stats import (
    compute_market_stats_from_store as _compute_market_stats_from_store_impl,
)
from quant_platform.engines.session.runtime import (
    hydrate_session_state as _hydrate_session_state,
)
from quant_platform.engines.session.runtime import (
    model_registry_preflight as _model_registry_preflight,
)
from quant_platform.engines.session.strategy_cycle import (
    durable_kill_switch as _durable_kill_switch_impl,
)
from quant_platform.engines.session.strategy_cycle import (
    record_nav_snapshot as _record_nav_snapshot_impl,
)
from quant_platform.engines.session.strategy_cycle import (
    run_strategy_cycle as _run_strategy_cycle_impl,
)
from quant_platform.engines.session.strategy_cycle import (
    run_strategy_cycle_unlocked as _run_strategy_cycle_unlocked_impl,
)
from quant_platform.engines.session.strategy_cycle import (
    strategy_cycle_lock as _strategy_cycle_lock_impl,
)
from quant_platform.infrastructure.support.distributed_lock import create_distributed_lock

if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.application.runtime.state import CycleResult, Session
    from quant_platform.core.domain.research import StrategyRun
    from quant_platform.core.domain.signals import RegimeState
    from quant_platform.services.signal_service.regime_detector import MarketStats

__all__ = [
    "hydrate_session_state",
    "model_registry_preflight",
    "run_strategy_cycle",
]


async def _compute_market_stats_from_store(
    session: Session,
    as_of: datetime,
) -> MarketStats | None:
    """Compatibility hook for tests and engine code patching session internals."""
    return await _compute_market_stats_from_store_impl(session, as_of)


async def _durable_kill_switch(policy: object, reason: str, activated_by: str) -> None:
    """Compatibility hook for durable kill-switch activation."""
    await _durable_kill_switch_impl(policy, reason, activated_by)


async def _run_strategy_cycle_unlocked(
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
    strategy_run: StrategyRun,
    market_prices: dict[uuid.UUID, Decimal] | None = None,
    regime: RegimeState | None = None,
    as_of: datetime | None = None,
    lock: object | None = None,
) -> CycleResult:
    """Compatibility wrapper around the runtime strategy-cycle use case."""
    return await _run_strategy_cycle_unlocked_impl(
        session=session,
        feature_data=feature_data,
        strategy_run=strategy_run,
        market_prices=market_prices,
        regime=regime,
        as_of=as_of,
        lock=lock,
        market_stats_reader=_compute_market_stats_from_store,
    )


@asynccontextmanager
async def _strategy_cycle_lock(
    session: Session,
    strategy_run_id: uuid.UUID,
) -> AsyncIterator[object]:
    """Compatibility wrapper around the runtime strategy-cycle lock."""
    async with _strategy_cycle_lock_impl(
        session,
        strategy_run_id,
        lock_factory=create_distributed_lock,
    ) as lock:
        yield lock


async def hydrate_session_state(session: Session) -> None:
    """Restore durable runtime state from backing stores."""
    await _hydrate_session_state(session)


async def model_registry_preflight(
    session: Session,
    *,
    strategy_name: str,
    engine_version: str,
    require_match: bool | None = None,
) -> None:
    """Compare the active ``RegisteredModel`` against the running engine."""
    await _model_registry_preflight(
        session,
        strategy_name=strategy_name,
        engine_version=engine_version,
        require_match=require_match,
    )


async def run_strategy_cycle(
    session: Session,
    feature_data: dict[uuid.UUID, dict[str, float]],
    strategy_run: StrategyRun,
    market_prices: dict[uuid.UUID, Decimal] | None = None,
    regime: RegimeState | None = None,
    as_of: datetime | None = None,
) -> CycleResult:
    """Execute one complete rebalance cycle with optional distributed locking."""
    return await _run_strategy_cycle_impl(
        session=session,
        feature_data=feature_data,
        strategy_run=strategy_run,
        market_prices=market_prices,
        regime=regime,
        as_of=as_of,
        cycle_runner=_run_strategy_cycle_unlocked,
        hydrator=hydrate_session_state,
        kill_switcher=_durable_kill_switch,
        nav_snapshot_recorder=_record_nav_snapshot,
        lock_context=_strategy_cycle_lock,
    )


async def _record_nav_snapshot(session: Session, strategy_run_id: uuid.UUID) -> None:
    """Persist a best-effort NAV snapshot for operator lifecycle metrics."""
    await _record_nav_snapshot_impl(session, strategy_run_id)
