"""Finalization and artifact writing for simple backtest runs."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.research import BacktestRun, StrategyRun
from quant_platform.services.research_service.backtesting.artifacts.backtest_artifact_io import (
    write_backtest_artifacts,
    write_execution_quality,
    write_run_summary,
)

from ..artifacts.backtest_artifacts import (
    BacktestCycleMetrics,
    BacktestFillArtifact,
    _compute_annualised_sharpe,
)
from ..simple.backtest_performance import (
    gross_turnover_from_fills,
    max_drawdown_from_nav,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import BacktestReplayBroker

log = structlog.get_logger(__name__)


async def finalize_backtest_run(
    *,
    settings: PlatformSettings,
    broker: BacktestReplayBroker,
    strategy_run: StrategyRun,
    start: datetime,
    end: datetime,
    initial_capital: Decimal,
    capital_snapshots: list[Decimal],
    cycle_metrics: list[BacktestCycleMetrics],
    fill_artifacts: list[BacktestFillArtifact],
    empty_timestamp: datetime,
    created_at: datetime,
    cumulative_commission: Decimal,
    cumulative_slippage_cost: Decimal,
    cumulative_slippage_bps: float,
) -> BacktestRun:
    """Persist backtest evidence and return the domain run summary."""
    final_account = await broker.sync_account()
    final_capital = final_account.net_asset_value
    total_return = (final_capital / initial_capital) - Decimal("1")

    artifact_uri = write_backtest_artifacts(
        settings.storage.object_store_root,
        strategy_run_id=strategy_run.run_id,
        cycle_metrics=cycle_metrics,
        fill_artifacts=fill_artifacts,
        empty_timestamp=empty_timestamp,
    )
    gross_turnover = gross_turnover_from_fills(fill_artifacts, initial_capital)
    await broker.disconnect()

    max_drawdown = max_drawdown_from_nav(capital_snapshots)
    annualised_sharpe = _compute_annualised_sharpe(capital_snapshots)

    log.info(
        "backtest.run_with_data.complete",
        run_id=str(strategy_run.run_id),
        initial_capital=str(initial_capital),
        final_capital=str(final_capital),
        total_return=str(total_return),
        max_drawdown=str(max_drawdown),
        total_commissions=str(cumulative_commission),
        total_slippage_cost=str(cumulative_slippage_cost),
        total_slippage_bps=f"{cumulative_slippage_bps:.1f}",
        artifact_uri=artifact_uri,
    )

    write_run_summary(
        settings.storage.object_store_root,
        strategy_run.run_id,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        annualised_sharpe=annualised_sharpe,
        max_drawdown=max_drawdown,
        gross_turnover=gross_turnover,
        nav_curve=capital_snapshots,
    )
    write_execution_quality(
        settings.storage.object_store_root,
        strategy_run.run_id,
        fill_artifacts=fill_artifacts,
        cycle_metrics=cycle_metrics,
    )
    return BacktestRun(
        backtest_id=uuid.uuid4(),
        strategy_run_id=strategy_run.run_id,
        universe_snapshot_id=uuid.uuid4(),
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        annualised_sharpe=annualised_sharpe,
        max_drawdown=max_drawdown,
        artifact_uri=artifact_uri,
        created_at=created_at,
    )
