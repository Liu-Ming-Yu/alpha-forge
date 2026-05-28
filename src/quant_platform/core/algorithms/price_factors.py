"""Pure price-series factor calculations shared across services."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class InsufficientDataError(ValueError):
    """Raised when a price series is too short to compute a factor."""


def sma(closes: Sequence[float], window: int) -> float:
    """Simple moving average of the last ``window`` close prices."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if len(closes) < window:
        raise InsufficientDataError(f"sma(window={window}) needs {window} bars, got {len(closes)}")
    return sum(closes[-window:]) / window


def trailing_log_returns(closes: Sequence[float], window: int, *, caller: str) -> list[float]:
    """Return the trailing ``window`` log returns with shared validation."""
    required = window + 1
    if len(closes) < required:
        raise InsufficientDataError(
            f"{caller}(window={window}) needs {required} bars, got {len(closes)}"
        )
    return [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - window, len(closes))]


def distance_to_52w_high(
    closes: Sequence[float],
    window: int = 252,
) -> float:
    """Distance of the latest close from the trailing high."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if len(closes) < window:
        raise InsufficientDataError(
            f"distance_to_52w_high(window={window}) needs {window} bars, got {len(closes)}"
        )
    trailing_high = max(closes[-window:])
    if trailing_high == 0.0:
        raise ValueError("distance_to_52w_high: trailing high is zero")
    return (closes[-1] - trailing_high) / trailing_high


def trend_z_score(closes: Sequence[float], window: int = 200) -> float:
    """Price deviation from its simple moving average as a fraction of the SMA."""
    moving_average = sma(closes, window)
    if moving_average == 0.0:
        return 0.0
    return (closes[-1] - moving_average) / moving_average


def realized_vol(
    closes: Sequence[float],
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Realized volatility from trailing log daily returns."""
    log_returns = trailing_log_returns(closes, window, caller="realized_vol")
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(trading_days) if annualize else daily_vol


def vol_compression_ratio(
    closes: Sequence[float],
    short_window: int = 5,
    long_window: int = 21,
) -> float:
    """Volatility compression ratio: short-term vol divided by long-term vol."""
    if long_window <= short_window:
        raise ValueError(
            f"long_window ({long_window}) must be greater than short_window ({short_window})"
        )
    short_vol = realized_vol(closes, short_window, annualize=False)
    long_vol = realized_vol(closes, long_window, annualize=False)
    if long_vol == 0.0:
        return 1.0
    return short_vol / long_vol


def momentum_return(closes: Sequence[float], period: int) -> float:
    """Simple price return over the last ``period`` trading days."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    required = period + 1
    if len(closes) < required:
        raise InsufficientDataError(
            f"momentum_return(period={period}) needs {required} bars, got {len(closes)}"
        )
    start = closes[-(period + 1)]
    end = closes[-1]
    if start == 0.0:
        raise ValueError("momentum_return: start price is zero")
    if end == 0.0:
        raise InsufficientDataError(
            f"momentum_return(period={period}): end price is zero (delisting-)"
        )
    return end / start - 1.0


def momentum_skip1m(
    closes: Sequence[float],
    long_period: int = 252,
    skip_period: int = 21,
) -> float:
    """12M-skip-1M momentum that avoids the most recent reversal window."""
    if long_period <= skip_period:
        raise ValueError(
            f"long_period ({long_period}) must be greater than skip_period ({skip_period})"
        )
    required = long_period + 1
    if len(closes) < required:
        raise InsufficientDataError(
            f"momentum_skip1m(long={long_period}, skip={skip_period}) needs "
            f"{required} bars, got {len(closes)}"
        )
    start = closes[-(long_period + 1)]
    end = closes[-(skip_period + 1)]
    if start == 0.0:
        raise ValueError("momentum_skip1m: start price is zero")
    if end == 0.0:
        raise InsufficientDataError(
            f"momentum_skip1m(long={long_period}, skip={skip_period}): "
            "end price is zero (delisting-)"
        )
    return end / start - 1.0


def short_term_reversal(closes: Sequence[float], window: int = 5) -> float:
    """Short-term reversal factor: negative of the cumulative trailing return."""
    return -momentum_return(closes, period=window)


def trend_quality(
    closes: Sequence[float],
    window: int = 63,
    eps: float = 1e-9,
) -> float:
    """Risk-adjusted momentum: mean trailing log return divided by volatility."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    log_returns = trailing_log_returns(closes, window, caller="trend_quality")
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    std = math.sqrt(variance)
    return mean / (std + eps)


def low_volatility(closes: Sequence[float], window: int = 63) -> float:
    """Low-volatility alpha: the negated annualized realized volatility.

    Calmer names score higher.  Captures the low-volatility anomaly — lower-risk
    stocks earn higher risk-adjusted returns — and is structurally
    anti-correlated with a crowded momentum book, so it diversifies the factor
    blend rather than echoing it.
    """
    return -realized_vol(closes, window, annualize=True)


def mean_reversion(closes: Sequence[float], window: int = 63) -> float:
    """Mean-reversion alpha: the negated deviation of price from its SMA.

    Names trading below their moving average score high (expected to revert up);
    extended names score low.  A medium-horizon counterweight to momentum.
    """
    return -trend_z_score(closes, window)


__all__ = [
    "InsufficientDataError",
    "distance_to_52w_high",
    "low_volatility",
    "mean_reversion",
    "momentum_return",
    "momentum_skip1m",
    "realized_vol",
    "short_term_reversal",
    "sma",
    "trailing_log_returns",
    "trend_quality",
    "trend_z_score",
    "vol_compression_ratio",
]
