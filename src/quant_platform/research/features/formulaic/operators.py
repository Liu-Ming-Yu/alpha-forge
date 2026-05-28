"""Operator library for the formulaic alpha factory.

The brief lists 20+ candidate operators across three axes. This module
ships **13 core operators** chosen to (a) cover all three axes, (b)
support the starter library in :mod:`.library`, and (c) leave the rest
of the WorldQuant catalog reachable from straightforward additions.

Categories
----------

* **Time-series** (per-instrument, sees history): ``delta``, ``delay``,
  ``ts_rank``, ``ts_zscore``, ``ts_corr``, ``decay_linear``,
  ``ts_argmax``. ``rolling_*`` aggregations are exposed for AST use too
  but delegate to :mod:`..transforms`.

* **Cross-sectional** (per-date, sees the universe): ``rank``,
  ``zscore``, ``group_rank``.

* **Element-wise** (per-row): ``abs``, ``sign``, ``signed_power``.

Each operator is registered into :data:`OPERATORS` keyed by name. The
:mod:`.evaluator` dispatches :class:`~.ast.OpCall` nodes through this
registry; the AST itself only carries the operator's name + lookback
contribution, never the function reference, so an alpha can be
serialised, mutated by a search procedure, or replayed against an
older operator-set version without picking up new behaviour
silently.

Public builders (``delta``, ``rank``, …) are thin convenience
constructors that build the correct :class:`OpCall` with the right
metadata. Library authors should always use the builders rather than
constructing ``OpCall("delta", ...)`` by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from quant_platform.research.features.formulaic.ast import (
    Expression,
    OpCall,
)
from quant_platform.research.features.transforms import safe_div

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.research.features.formulaic.panel import MarketPanel

Axis = Literal["time_series", "cross_sectional", "element_wise"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_by_instrument(
    panel: MarketPanel, series: pd.Series
) -> pd.core.groupby.generic.SeriesGroupBy:
    """Return a per-instrument GroupBy view over ``series``.

    ``series`` is index-aligned to ``panel.frame``; pandas allows
    grouping a Series by another Series of the same length, so the
    grouper is just the panel's ``instrument_id`` column.
    """
    return series.groupby(panel.frame["instrument_id"], sort=False, group_keys=False)


def _group_by_date(
    panel: MarketPanel,
    series: pd.Series,
    *,
    date_column: str = "date",
) -> pd.core.groupby.generic.SeriesGroupBy:
    """Return a per-date GroupBy view over ``series``."""
    return series.groupby(panel.frame[date_column], sort=False, group_keys=False)


# ---------------------------------------------------------------------------
# Operator metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operator:
    """Static metadata about an operator (the compute function is registered
    separately so the dataclass stays freely-hashable)."""

    name: str
    axis: Axis
    description: str


# ---------------------------------------------------------------------------
# Time-series operators (per-instrument)
# ---------------------------------------------------------------------------


def _compute_delta(panel: MarketPanel, x: pd.Series, periods: int) -> pd.Series:
    grouped = _group_by_instrument(panel, x)
    return x - grouped.shift(periods)


def delta(x: Expression, periods: int) -> OpCall:
    """``delta(x, d) = x - x.shift(d)``, per instrument."""
    return OpCall(
        name="delta",
        args=(x,),
        window_lookback=periods,
        int_args=(periods,),
    )


def _compute_delay(panel: MarketPanel, x: pd.Series, periods: int) -> pd.Series:
    return _group_by_instrument(panel, x).shift(periods)


def delay(x: Expression, periods: int) -> OpCall:
    """``delay(x, d) = x.shift(d)``, per instrument."""
    return OpCall(
        name="delay",
        args=(x,),
        window_lookback=periods,
        int_args=(periods,),
    )


def _compute_ts_rank(panel: MarketPanel, x: pd.Series, window: int) -> pd.Series:
    grouped = _group_by_instrument(panel, x)
    return grouped.transform(
        lambda s: s.rolling(window, min_periods=window).rank(method="average", pct=True)
    )


def ts_rank(x: Expression, window: int) -> OpCall:
    """Rolling rank-pct of ``x`` over ``window`` rows, per instrument."""
    return OpCall(
        name="ts_rank",
        args=(x,),
        window_lookback=window,
        int_args=(window,),
    )


def _compute_ts_zscore(panel: MarketPanel, x: pd.Series, window: int) -> pd.Series:
    grouped = _group_by_instrument(panel, x)
    mean = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
    std = grouped.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=1))
    return safe_div(x - mean, std, require_positive_denom=False)


def ts_zscore(x: Expression, window: int) -> OpCall:
    """Per-instrument rolling z-score of ``x`` over ``window`` rows."""
    return OpCall(
        name="ts_zscore",
        args=(x,),
        window_lookback=window,
        int_args=(window,),
    )


def _compute_ts_corr(
    panel: MarketPanel,
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> pd.Series:
    grouper = panel.frame["instrument_id"]
    # Per-instrument rolling Pearson correlation. pandas' ``rolling.corr``
    # on a SeriesGroupBy returns a multi-index frame, so we route through
    # a per-group apply + re-flatten path.
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for _, idx in x.groupby(grouper, sort=False).groups.items():
        xi = x.loc[idx]
        yi = y.loc[idx]
        rolled = xi.rolling(window, min_periods=window).corr(yi)
        out.loc[idx] = rolled.to_numpy()
    return out


def ts_corr(x: Expression, y: Expression, window: int) -> OpCall:
    """Per-instrument rolling Pearson correlation of ``x`` and ``y``."""
    return OpCall(
        name="ts_corr",
        args=(x, y),
        window_lookback=window,
        int_args=(window,),
    )


def _compute_decay_linear(panel: MarketPanel, x: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    grouped = _group_by_instrument(panel, x)

    def _weighted(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=window).apply(
            lambda arr: float(np.dot(arr, weights)),
            raw=True,
        )

    return grouped.transform(_weighted)


def decay_linear(x: Expression, window: int) -> OpCall:
    """Linearly-decaying weighted average of ``x`` over ``window`` rows.

    Weights are ``[1, 2, …, window]`` normalised to sum to 1, so the
    most recent observation gets the highest weight. Per-instrument.
    """
    return OpCall(
        name="decay_linear",
        args=(x,),
        window_lookback=window,
        int_args=(window,),
    )


def _compute_ts_argmax(panel: MarketPanel, x: pd.Series, window: int) -> pd.Series:
    """Position (1..window) of the max value inside the trailing window.

    Returns ``1`` when the most-recent value is the max, ``window`` when
    the oldest value in the window is the max. Per-instrument.
    """
    grouped = _group_by_instrument(panel, x)
    return grouped.transform(
        lambda s: s.rolling(window, min_periods=window).apply(
            lambda arr: float(window - int(np.argmax(arr))),
            raw=True,
        )
    )


def ts_argmax(x: Expression, window: int) -> OpCall:
    """Position of the rolling-window max, per instrument."""
    return OpCall(
        name="ts_argmax",
        args=(x,),
        window_lookback=window,
        int_args=(window,),
    )


# ---------------------------------------------------------------------------
# Cross-sectional operators (per-date)
# ---------------------------------------------------------------------------


def _compute_rank(panel: MarketPanel, x: pd.Series) -> pd.Series:
    return _group_by_date(panel, x).transform(lambda s: s.rank(method="average", pct=True))


def rank(x: Expression) -> OpCall:
    """Per-date rank-pct of ``x`` (range ``[0, 1]``)."""
    return OpCall(name="rank", args=(x,))


def _compute_zscore(panel: MarketPanel, x: pd.Series) -> pd.Series:
    grouped = _group_by_date(panel, x)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda s: s.std(ddof=0))
    return safe_div(x - mean, std, require_positive_denom=False)


def zscore(x: Expression) -> OpCall:
    """Per-date population z-score of ``x``."""
    return OpCall(name="zscore", args=(x,))


def _compute_group_rank(
    panel: MarketPanel,
    x: pd.Series,
    group_column: str,
) -> pd.Series:
    panel.require_column(group_column)
    grouper = [panel.frame["date"], panel.frame[group_column]]
    return x.groupby(grouper, sort=False, group_keys=False).transform(
        lambda s: s.rank(method="average", pct=True)
    )


def group_rank(x: Expression, group_column: str) -> OpCall:
    """Per-(date, ``group_column``) rank-pct.

    Common use: ``group_rank(close, "sector")`` for sector-relative
    pricing.
    """
    return OpCall(name="group_rank", args=(x,), str_args=(group_column,))


# ---------------------------------------------------------------------------
# Element-wise operators
# ---------------------------------------------------------------------------


def _compute_abs(panel: MarketPanel, x: pd.Series) -> pd.Series:
    return x.abs()


def absolute(x: Expression) -> OpCall:
    """Element-wise ``abs(x)``.

    Builder is named ``absolute`` to avoid shadowing the Python
    built-in; the operator's registry key is still ``"abs"``.
    """
    return OpCall(name="abs", args=(x,))


def _compute_sign(panel: MarketPanel, x: pd.Series) -> pd.Series:
    return np.sign(x).astype(float)


def sign(x: Expression) -> OpCall:
    """Element-wise ``sign(x)``: -1, 0, or +1."""
    return OpCall(name="sign", args=(x,))


def _compute_signed_power(panel: MarketPanel, x: pd.Series, exponent: float) -> pd.Series:
    return np.sign(x) * np.abs(x).pow(exponent)


def signed_power(x: Expression, exponent: float) -> OpCall:
    """Element-wise ``sign(x) * |x| ** exponent``.

    Preserves the sign of ``x`` while applying a magnitude transform.
    Common WorldQuant idiom for compressing tails.
    """
    return OpCall(name="signed_power", args=(x,), float_args=(exponent,))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Per-operator metadata + compute function. The compute function's
#: signature follows the convention
#: ``(panel, *evaluated_series_args, *int_args, *str_args, *float_args)``.
#: The evaluator unpacks the OpCall and dispatches; operator authors
#: don't write to this dict directly — they edit the builders above
#: and add an entry here.
ComputeFn = Callable[..., pd.Series]
OPERATORS: Mapping[str, tuple[Operator, ComputeFn]] = {
    "delta": (Operator("delta", "time_series", "x - x.shift(d)"), _compute_delta),
    "delay": (Operator("delay", "time_series", "x.shift(d)"), _compute_delay),
    "ts_rank": (
        Operator("ts_rank", "time_series", "rolling rank-pct over d periods"),
        _compute_ts_rank,
    ),
    "ts_zscore": (
        Operator("ts_zscore", "time_series", "rolling z-score over d periods"),
        _compute_ts_zscore,
    ),
    "ts_corr": (
        Operator("ts_corr", "time_series", "rolling Pearson correlation of x, y over d periods"),
        _compute_ts_corr,
    ),
    "decay_linear": (
        Operator(
            "decay_linear",
            "time_series",
            "linearly-weighted moving average; most-recent gets highest weight",
        ),
        _compute_decay_linear,
    ),
    "ts_argmax": (
        Operator("ts_argmax", "time_series", "position of rolling-window max in [1..d]"),
        _compute_ts_argmax,
    ),
    "rank": (
        Operator("rank", "cross_sectional", "per-date rank-pct over the universe"),
        _compute_rank,
    ),
    "zscore": (
        Operator("zscore", "cross_sectional", "per-date population z-score"),
        _compute_zscore,
    ),
    "group_rank": (
        Operator("group_rank", "cross_sectional", "per-(date, group) rank-pct"),
        _compute_group_rank,
    ),
    "abs": (Operator("abs", "element_wise", "element-wise absolute value"), _compute_abs),
    "sign": (Operator("sign", "element_wise", "element-wise sign in {-1, 0, +1}"), _compute_sign),
    "signed_power": (
        Operator(
            "signed_power",
            "element_wise",
            "sign(x) * |x|^p — preserves sign, transforms magnitude",
        ),
        _compute_signed_power,
    ),
}


def dispatch(op_call: OpCall) -> tuple[Operator, ComputeFn]:
    """Look up the operator + compute function for an :class:`OpCall`.

    Raises :class:`KeyError` (with a known-operator list in the message)
    when ``op_call.name`` is not registered.
    """
    try:
        return OPERATORS[op_call.name]
    except KeyError as exc:
        known = sorted(OPERATORS)
        raise KeyError(
            f"formulaic.operators: unknown operator {op_call.name!r}; known operators: {known!r}"
        ) from exc


__all__ = [
    "OPERATORS",
    "Axis",
    "ComputeFn",
    "Operator",
    "absolute",
    "decay_linear",
    "delay",
    "delta",
    "dispatch",
    "group_rank",
    "rank",
    "sign",
    "signed_power",
    "ts_argmax",
    "ts_corr",
    "ts_rank",
    "ts_zscore",
    "zscore",
]
