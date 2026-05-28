"""Pure return and equity-curve metrics for research campaigns."""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.constants import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Sequence


def sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    if std <= 0:
        return 0.0
    return (statistics.mean(returns) / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def compound_return(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    high = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        high = max(high, equity)
        if high > 0:
            worst = min(worst, (equity - high) / high)
    return worst


def equity_curve(returns: Sequence[float]) -> list[float]:
    equity = Decimal("1")
    curve = [float(equity)]
    for value in returns:
        equity *= Decimal(str(1.0 + value))
        curve.append(float(equity))
    return curve


def non_overlapping_bucket_returns(
    daily_returns: Sequence[float],
    horizon_days: int,
) -> list[float]:
    """Compound daily simple returns into non-overlapping bucket returns.

    Used to derive the diagnostic ``bucket_<H>d`` return series from the
    canonical daily mark-to-market stream. A bucket of ``H`` daily simple
    returns ``r_0..r_{H-1}`` produces ``prod(1 + r_i) - 1``.

    Partial trailing buckets (fewer than ``horizon_days`` observations) are
    discarded — including them would understate the variance of the bucket
    series and inflate the bucket Sharpe.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    buckets: list[float] = []
    n = len(daily_returns)
    for start in range(0, n - horizon_days + 1, horizon_days):
        equity = 1.0
        for value in daily_returns[start : start + horizon_days]:
            equity *= 1.0 + value
        buckets.append(equity - 1.0)
    return buckets


def bucket_sharpe(bucket_returns: Sequence[float], horizon_days: int) -> float:
    """Annualized Sharpe for non-overlapping bucket returns of horizon H.

    Annualization factor is ``sqrt(TRADING_DAYS_PER_YEAR / H)`` rather
    than ``sqrt(TRADING_DAYS_PER_YEAR)`` because each observation covers
    ``H`` trading days, not one.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if len(bucket_returns) < 2:
        return 0.0
    std = statistics.stdev(bucket_returns)
    if std <= 0:
        return 0.0
    return (statistics.mean(bucket_returns) / std) * math.sqrt(TRADING_DAYS_PER_YEAR / horizon_days)


__all__ = [
    "bucket_sharpe",
    "compound_return",
    "equity_curve",
    "max_drawdown",
    "non_overlapping_bucket_returns",
    "sharpe",
]
