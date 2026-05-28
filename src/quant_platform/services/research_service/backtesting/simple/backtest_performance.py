"""Pure performance accounting helpers for research backtests."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from ..artifacts.backtest_artifacts import (
        BacktestFillArtifact,
    )


def gross_turnover_from_fills(
    fill_artifacts: list[BacktestFillArtifact],
    initial_capital: Decimal,
) -> Decimal:
    if initial_capital <= 0 or not fill_artifacts:
        return Decimal("0")
    total_notional = sum(
        abs(Decimal(str(fill.adjusted_fill_price)) * Decimal(fill.quantity))
        for fill in fill_artifacts
    )
    return (total_notional / initial_capital).quantize(Decimal("0.0001"))


def max_drawdown_from_nav(
    nav_curve: list[Decimal], initial_capital: Decimal | None = None
) -> Decimal:
    if not nav_curve:
        return Decimal("0")
    peak = initial_capital if initial_capital is not None else nav_curve[0]
    worst = Decimal("0")
    for nav in nav_curve:
        if nav > peak:
            peak = nav
        drawdown = (nav - peak) / peak if peak != 0 else Decimal("0")
        if drawdown < worst:
            worst = drawdown
    return worst


def periods_per_year_from_rebalances(rebalance_timestamps: list[datetime]) -> float:
    if len(rebalance_timestamps) < 2:
        return 252.0
    total_days = (rebalance_timestamps[-1] - rebalance_timestamps[0]).days
    n_periods = len(rebalance_timestamps) - 1
    avg_period_days = total_days / n_periods if n_periods > 0 else 1.0
    return 365.0 / avg_period_days if avg_period_days > 0 else 252.0
