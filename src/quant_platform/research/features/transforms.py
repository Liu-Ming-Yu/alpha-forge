"""Shared, point-in-time-safe building blocks for feature families.

Every helper here follows two non-negotiable rules:

1. Rolling and shift operations are applied within an instrument group.
2. Rolling windows require a full lookback by default, so warm-up rows remain
   ``NaN`` unless a caller explicitly asks for partial windows.

The module also publishes a small set of **tokens** (calendar constants,
sentinels, default key columns) that family modules should import rather
than hard-code. Adding a new family means importing from this file; never
re-derive these values in family-local code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
#
# Pre-existing magic literals promoted to named constants so every family
# uses the same value. Renaming or changing any of these is a cross-family
# API decision — touch it once, here.

# Canonical financial constants live in ``quant_platform.core.constants``
# so cross-layer code (services, infrastructure) can import them without
# crossing the architecture boundary that forbids ``services → research``
# imports. ``features/transforms.py`` re-exports them here so existing
# feature-family code keeps its single import site.
from quant_platform.core.constants import (
    BPS_PER_UNIT,
    CALENDAR_DAYS_PER_QUARTER,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_YEAR,
)

#: Sentinel group used when neutralising a feature against a categorical
#: covariate whose value is missing (e.g. an instrument with no sector
#: mapping). Putting the rows in a named bucket keeps them in the
#: groupby aggregation rather than silently dropping them.
UNKNOWN_GROUP_SENTINEL: str = "__unknown__"

#: Canonical key columns for a daily-bar feature panel. Fundamentals
#: panels override with ``("instrument_id", "datekey")``; new families
#: should reuse whichever already matches their grain rather than
#: invent a third spelling.
DEFAULT_KEY_COLUMNS: tuple[str, ...] = ("instrument_id", "date")
KEY_COLUMNS_FUNDAMENTALS: tuple[str, ...] = ("instrument_id", "datekey")


# ---------------------------------------------------------------------------
# Rolling-window policy
# ---------------------------------------------------------------------------

MinPeriodsPolicy = Literal["full", "partial"]


def _resolve_min_periods(window: int, policy: MinPeriodsPolicy) -> int:
    """Translate a rolling-window policy into a ``min_periods`` value."""
    if window <= 0:
        raise ValueError("rolling window must be > 0")
    if policy == "full":
        return window
    if policy == "partial":
        return 1
    raise ValueError(f"unknown min_periods policy: {policy!r}")


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def safe_div(
    numer: pd.Series,
    denom: pd.Series,
    *,
    require_positive_denom: bool = True,
    min_abs_denom: float | None = None,
) -> pd.Series:
    """Element-wise divide with an explicit denominator policy.

    By default, zero, negative, and missing denominators produce ``NaN``. Some
    feature families may deliberately allow negative denominators; they should
    opt in with ``require_positive_denom=False`` at the call site.
    """
    out = numer / denom
    mask = denom.notna() & (denom != 0)

    if require_positive_denom:
        mask &= denom > 0
    if min_abs_denom is not None:
        if min_abs_denom < 0:
            raise ValueError("min_abs_denom must be >= 0 when provided")
        mask &= denom.abs() >= min_abs_denom

    return out.where(mask, np.nan)


# ---------------------------------------------------------------------------
# Group-by-instrument helpers
# ---------------------------------------------------------------------------
#
# ``sort=False`` preserves the first-seen instrument order so the rolling
# output is row-aligned with the (already-sorted by ``(instrument_id, date)``)
# input. ``group_keys=False`` keeps the output Series flat instead of
# multiindex-prefixed by group key.


def group_by_instrument(
    df: pd.DataFrame,
    *,
    instrument_column: str = "instrument_id",
) -> DataFrameGroupBy:
    """Return a stable per-instrument GroupBy view over ``df``.

    Centralised so every family uses the same ``sort=False`` /
    ``group_keys=False`` flags; without this, contributors copy the
    ``groupby`` line from whichever family they read first and the
    flag conventions drift.
    """
    return df.groupby(instrument_column, sort=False, group_keys=False)


def group_rolling_mean(
    grouped: SeriesGroupBy,
    window: int,
    *,
    policy: MinPeriodsPolicy = "full",
) -> pd.Series:
    """Per-instrument rolling mean over ``window`` rows."""
    min_periods = _resolve_min_periods(window, policy)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())


def group_rolling_std(
    grouped: SeriesGroupBy,
    window: int,
    *,
    policy: MinPeriodsPolicy = "full",
    ddof: int = 1,
) -> pd.Series:
    """Per-instrument rolling standard deviation over ``window`` rows."""
    min_periods = _resolve_min_periods(window, policy)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=ddof))


def group_rolling_sum(
    grouped: SeriesGroupBy,
    window: int,
    *,
    policy: MinPeriodsPolicy = "full",
) -> pd.Series:
    """Per-instrument rolling sum over ``window`` rows."""
    min_periods = _resolve_min_periods(window, policy)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).sum())


def group_rolling_max(
    grouped: SeriesGroupBy,
    window: int,
    *,
    policy: MinPeriodsPolicy = "full",
) -> pd.Series:
    """Per-instrument rolling max over ``window`` rows."""
    min_periods = _resolve_min_periods(window, policy)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).max())


def group_rolling_min(
    grouped: SeriesGroupBy,
    window: int,
    *,
    policy: MinPeriodsPolicy = "full",
) -> pd.Series:
    """Per-instrument rolling min over ``window`` rows."""
    min_periods = _resolve_min_periods(window, policy)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).min())


def group_shift(grouped: SeriesGroupBy, periods: int) -> pd.Series:
    """Per-instrument shift that respects group boundaries."""
    return grouped.shift(periods)


def group_pct_change(grouped: SeriesGroupBy, periods: int) -> pd.Series:
    """Per-instrument percent change over ``periods`` rows."""
    current = grouped.obj
    lagged = grouped.shift(periods)
    return safe_div(current, lagged) - 1.0


# ---------------------------------------------------------------------------
# Series coercion helpers
# ---------------------------------------------------------------------------


def ones_like(series: pd.Series) -> pd.Series:
    """Return a float Series of ones with ``series``'s index.

    Useful for building reciprocal features (``1 / pe``, ``1 / pb``)
    via :func:`safe_div` without inlining a ``pd.Series(1.0, ...)``
    constant at every call site.
    """
    return pd.Series(1.0, index=series.index, dtype=float)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce ``series`` to float, replacing non-numeric sentinels with NaN.

    Some vendor columns arrive with ``object`` dtype because the upstream
    loader did not normalise blanks and ``--``-style sentinels into NaN.
    Use this at the family boundary so all downstream math sees floats.
    """
    return pd.to_numeric(series, errors="coerce").astype(float)


__all__ = [
    "BPS_PER_UNIT",
    "CALENDAR_DAYS_PER_QUARTER",
    "DEFAULT_KEY_COLUMNS",
    "KEY_COLUMNS_FUNDAMENTALS",
    "MinPeriodsPolicy",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_YEAR",
    "UNKNOWN_GROUP_SENTINEL",
    "coerce_numeric",
    "group_by_instrument",
    "group_pct_change",
    "group_rolling_max",
    "group_rolling_mean",
    "group_rolling_min",
    "group_rolling_std",
    "group_rolling_sum",
    "group_shift",
    "ones_like",
    "safe_div",
]
