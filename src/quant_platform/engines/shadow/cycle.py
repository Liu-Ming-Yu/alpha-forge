"""Shadow-mode target-building cycle for engine runners."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.runtime.state import CycleResult
from quant_platform.core.domain.production import NavSnapshot
from quant_platform.engines.proposals.target_pipeline import build_engine_target

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research.runs import StrategyRun
log = structlog.get_logger(__name__)


async def run_shadow_target_cycle(
    *,
    session: Session,
    strategy_run: StrategyRun,
    feature_data: dict[uuid.UUID, dict[str, float]],
    market_prices: Mapping[uuid.UUID, Decimal] | None = None,
    engine_name: str,
) -> CycleResult:
    """Generate shadow signals, target, and approvals without submitting."""
    if (
        session.signal_ctrl is None
        or session.portfolio_ctrl is None
        or session.order_planner is None
        or session.regime_detector is None
    ):
        raise RuntimeError("shadow target session is missing required controllers")

    as_of = session.clock.now()
    prices = dict(market_prices or {})
    target_build = await build_engine_target(
        session=session,
        strategy_run=strategy_run,
        feature_data=feature_data,
        as_of=as_of,
    )

    intents: list[OrderIntent] = []
    approved: list[OrderIntent] = []
    rejected: list[OrderIntent] = []
    if target_build.target is not None:
        intents = session.order_planner.plan(
            target=target_build.target,
            account=target_build.account,
            market_prices=prices,
            strategy_run_id=strategy_run.run_id,
        )
        approved, rejected = await session.approve_ctrl.approve(
            intents,
            target_build.account,
        )

    log.info(
        "engine_runner.shadow_cycle",
        signals=len(target_build.signals),
        target_weights=len(target_build.target.weights) if target_build.target else 0,
        would_be_orders=len(intents),
        approved=len(approved),
        rejected=len(rejected),
        regime=target_build.regime.regime_label.value if target_build.regime else "unknown",
    )

    await save_shadow_nav_snapshot(
        session=session,
        strategy_run=strategy_run,
        account=target_build.account,
    )

    return CycleResult(
        signals=target_build.signals,
        target=target_build.target,
        approved=approved,
        rejected=rejected,
        submitted_ids=[],
        fills=[],
    )


async def save_shadow_nav_snapshot(
    *,
    session: Session,
    strategy_run: StrategyRun,
    account: AccountSnapshot,
) -> None:
    """Persist a shadow NAV snapshot without letting telemetry block the cycle."""
    try:
        gross_exposure = sum(
            (position.market_value for position in account.positions),
            Decimal("0"),
        )
        await session.performance_repo.save_nav_snapshot(
            NavSnapshot(
                snapshot_id=uuid.uuid4(),
                strategy_run_id=strategy_run.run_id,
                as_of=account.as_of,
                net_asset_value=account.net_asset_value,
                gross_exposure=gross_exposure,
                cash=account.settled_cash,
                source=account.source,
            )
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        log.warning("engine_runner.shadow_nav_snapshot_failed", error=str(exc))
