"""Result accounting helpers for VectorBT backtests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import BacktestRun, StrategyRun

from ..artifacts.backtest_artifacts import (
    _compute_annualised_sharpe,
)
from ..simple.backtest_performance import (
    max_drawdown_from_nav,
    periods_per_year_from_rebalances,
)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(slots=True)
class VectorBTBacktestSummary:
    """Lightweight intermediate result from the vectorized transform."""

    final_capital: Decimal
    total_return: Decimal
    annualised_sharpe: Decimal | None
    max_drawdown: Decimal
    equity_curve: list[Decimal]


def summarize_vectorbt_backtest(
    *,
    nav_curve: list[Decimal],
    initial_capital: Decimal,
    rebalance_timestamps: list[datetime],
) -> VectorBTBacktestSummary:
    final_capital = nav_curve[-1] if nav_curve else initial_capital
    total_return = (final_capital / initial_capital) - Decimal("1")
    annualised_sharpe = _compute_annualised_sharpe(
        nav_curve,
        periods_per_year=periods_per_year_from_rebalances(rebalance_timestamps),
    )
    max_drawdown = max_drawdown_from_nav(nav_curve, initial_capital=initial_capital)
    return VectorBTBacktestSummary(
        final_capital=final_capital,
        total_return=total_return,
        annualised_sharpe=annualised_sharpe,
        max_drawdown=max_drawdown,
        equity_curve=nav_curve,
    )


def build_vectorbt_backtest_run(
    *,
    strategy_run: StrategyRun,
    start: datetime,
    end: datetime,
    initial_capital: Decimal,
    summary: VectorBTBacktestSummary,
    artifact_uri: str,
    created_at: datetime,
) -> BacktestRun:
    return BacktestRun(
        backtest_id=uuid.uuid4(),
        strategy_run_id=strategy_run.run_id,
        universe_snapshot_id=uuid.uuid4(),
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        final_capital=summary.final_capital,
        total_return=summary.total_return,
        annualised_sharpe=summary.annualised_sharpe,
        max_drawdown=summary.max_drawdown,
        artifact_uri=artifact_uri,
        created_at=created_at,
    )
