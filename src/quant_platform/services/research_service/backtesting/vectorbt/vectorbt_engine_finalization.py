"""Finalization helpers for VectorBT backtest runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from .vectorbt_artifacts import (
    write_vectorbt_outputs,
)
from .vectorbt_results import (
    build_vectorbt_backtest_run,
    summarize_vectorbt_backtest,
)

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import Clock
    from quant_platform.core.domain.research import BacktestRun, StrategyRun

    from ..artifacts.backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )

log = structlog.get_logger(__name__)


def finalize_vectorbt_backtest_run(
    *,
    settings: PlatformSettings,
    clock: Clock,
    strategy_run: StrategyRun,
    start: datetime,
    end: datetime,
    initial_capital: Decimal,
    rebalance_timestamps: list[datetime],
    nav_curve: list[Decimal],
    fill_artifacts: list[BacktestFillArtifact],
    cycle_metrics: list[BacktestCycleMetrics],
) -> BacktestRun:
    """Write VectorBT artifacts and build the BacktestRun domain object."""
    summary = summarize_vectorbt_backtest(
        nav_curve=nav_curve,
        initial_capital=initial_capital,
        rebalance_timestamps=rebalance_timestamps,
    )
    artifact_uri = write_vectorbt_outputs(
        object_store_root=Path(settings.storage.object_store_root),
        strategy_run=strategy_run,
        initial_capital=initial_capital,
        final_capital=summary.final_capital,
        total_return=summary.total_return,
        annualised_sharpe=summary.annualised_sharpe,
        max_drawdown=summary.max_drawdown,
        nav_curve=summary.equity_curve,
        fill_artifacts=fill_artifacts,
        cycle_metrics=cycle_metrics,
        empty_timestamp=clock.now(),
    )
    log.info(
        "vectorbt_backtest.complete",
        run_id=str(strategy_run.run_id),
        initial_capital=str(initial_capital),
        final_capital=str(summary.final_capital),
    )
    return build_vectorbt_backtest_run(
        strategy_run=strategy_run,
        start=start,
        end=end,
        initial_capital=initial_capital,
        summary=summary,
        artifact_uri=artifact_uri,
        created_at=clock.now(),
    )
