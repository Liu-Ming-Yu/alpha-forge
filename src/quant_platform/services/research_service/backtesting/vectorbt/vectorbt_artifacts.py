"""Artifact persistence helpers for VectorBT backtests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.backtesting.artifacts.backtest_artifact_io import (
    write_backtest_artifacts,
    write_execution_quality,
    write_run_summary,
)

from ..simple.backtest_performance import (
    gross_turnover_from_fills,
)

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal
    from pathlib import Path

    from quant_platform.core.domain.research import StrategyRun

    from ..artifacts.backtest_artifacts import (
        BacktestCycleMetrics,
        BacktestFillArtifact,
    )


def write_vectorbt_outputs(
    *,
    object_store_root: Path,
    strategy_run: StrategyRun,
    initial_capital: Decimal,
    final_capital: Decimal,
    total_return: Decimal,
    annualised_sharpe: Decimal | None,
    max_drawdown: Decimal,
    nav_curve: list[Decimal],
    fill_artifacts: list[BacktestFillArtifact],
    cycle_metrics: list[BacktestCycleMetrics],
    empty_timestamp: datetime,
) -> str:
    """Write standard VectorBT backtest artifacts and return artifact URI."""
    artifact_uri = write_backtest_artifacts(
        object_store_root,
        strategy_run_id=strategy_run.run_id,
        cycle_metrics=cycle_metrics,
        fill_artifacts=fill_artifacts,
        empty_timestamp=empty_timestamp,
    )
    gross_turnover = gross_turnover_from_fills(fill_artifacts, initial_capital)
    write_run_summary(
        object_store_root,
        strategy_run.run_id,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        annualised_sharpe=annualised_sharpe,
        max_drawdown=max_drawdown,
        gross_turnover=gross_turnover,
        nav_curve=nav_curve,
    )
    write_execution_quality(
        object_store_root,
        strategy_run.run_id,
        fill_artifacts=fill_artifacts,
        cycle_metrics=cycle_metrics,
    )
    return artifact_uri
