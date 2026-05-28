"""Evidence computation for auto-discovered alpha candidates.

Every candidate the miner sees gets scored with a small panel of
metrics: information coefficient (Pearson and Spearman), ICIR
(Information-Coefficient Information-Ratio: per-date mean IC divided
by per-date std), turnover, coverage, and a worst-case correlation
against a set of already-admitted baseline features.

The contract is intentionally narrow: take an :class:`Expression`,
evaluate it against a :class:`MarketPanel` to get a feature Series,
and score that Series against a supplied forward-return label
Series. We do **not** wire into the full ``feature_governance``
walk-forward path here — that lives one floor up. This module's job
is to make a candidate scorable in a few hundred ms so the search
loop can chew through thousands of them.

Label-generation helper :func:`make_forward_return_labels` is exposed
so callers don't have to hand-derive forward returns at every call
site. Its output is index-aligned to the panel so the IC computation
can do a clean per-date groupby join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.transforms import (
    group_by_instrument,
    group_shift,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.panel import MarketPanel


@dataclass(frozen=True)
class CandidateEvidence:
    """Per-candidate evaluation result.

    Attributes
    ----------
    mean_ic:
        Per-date Pearson correlation between the candidate feature
        and the forward-return label, averaged over dates. ``NaN``
        if no date had enough valid pairs to compute IC.
    rank_ic:
        Per-date Spearman (rank) correlation, averaged. The brief's
        admission gates lean on rank IC because it's outlier-robust.
    icir:
        ``mean_ic / std(per_date_ic)``. Higher = more consistent
        signal across dates. ``NaN`` when std is zero or undefined.
    turnover:
        Per-date L1 turnover of the rank-normalised feature, averaged
        across dates. Roughly the fraction of the cross-section that
        changes its decile from one day to the next. ``NaN`` when
        only one date has data.
    coverage:
        Number of (instrument, date) rows where both the feature and
        the label are non-null. The miner's gate rejects candidates
        below a configured fraction of full coverage.
    correlation_to_baseline_max:
        Maximum (over baseline features) of the absolute pooled
        correlation between this candidate and each baseline. Used by
        :mod:`.admission` for diversity pruning. ``NaN`` when no
        baseline features were supplied.
    n_dates:
        Number of distinct dates that contributed to ``mean_ic``.
        Useful for filtering out candidates whose lookback ate the
        whole panel.
    """

    mean_ic: float
    rank_ic: float
    icir: float
    turnover: float
    coverage: int
    correlation_to_baseline_max: float
    n_dates: int


# ---------------------------------------------------------------------------
# Forward-return labels
# ---------------------------------------------------------------------------


def make_forward_return_labels(
    panel: MarketPanel,
    *,
    horizon: int = 5,
    price_column: str = "close",
) -> pd.Series:
    """Compute per-instrument forward total return over ``horizon`` days.

    For each ``(instrument_id, date)`` row, returns the simple total
    return from ``date`` to ``date + horizon`` trading days, computed
    against the panel's own dates (no calendar arithmetic). The last
    ``horizon`` rows of every instrument are ``NaN`` (no future data).

    The output is index-aligned to ``panel.frame`` so callers can
    pass it straight into :func:`compute_evidence`.
    """
    prices = panel.frame[price_column].astype(float)
    grouped = group_by_instrument(panel.frame)[price_column]
    future_price = group_shift(grouped, -horizon)
    # forward_return = future_price / current - 1
    return safe_div(future_price, prices, require_positive_denom=False) - 1.0


# ---------------------------------------------------------------------------
# IC + correlation primitives
# ---------------------------------------------------------------------------


def _per_date_correlation(
    panel: MarketPanel,
    feature: pd.Series,
    label: pd.Series,
    *,
    method: str = "pearson",
) -> pd.Series:
    """Per-date correlation between ``feature`` and ``label``.

    Returns a Series indexed by date. Dates with fewer than 2 valid
    pairs produce NaN. ``method`` is forwarded to
    :meth:`pandas.Series.corr` so both ``"pearson"`` and ``"spearman"``
    are accepted.

    The per-date loop is explicit rather than going through
    ``df.groupby(...).apply(...)`` because the latter returns a
    DataFrame on some pandas paths (when the group key is interpreted
    as a column) and a Series on others, depending on the pandas
    version and whether ``include_groups=False`` is recognised.
    Manual iteration produces a Series unambiguously.
    """
    df = pd.DataFrame(
        {
            "date": panel.frame["date"].to_numpy(),
            "feature": feature.to_numpy(),
            "label": label.to_numpy(),
        }
    )
    df = df.dropna(subset=["feature", "label"])
    if df.empty:
        return pd.Series(dtype=float)

    per_date: dict[object, float] = {}
    for date, group in df.groupby("date", sort=True):
        if len(group) < 2:
            per_date[date] = float("nan")
        else:
            per_date[date] = float(group["feature"].corr(group["label"], method=method))
    return pd.Series(per_date, dtype=float)


def _per_date_turnover(panel: MarketPanel, feature: pd.Series) -> pd.Series:
    """Per-date L1 turnover of the rank-normalised feature.

    Roughly: rank the cross-section to [0, 1] each date, take per-
    instrument absolute change from yesterday's rank, average over
    instruments. Captures how aggressively the alpha rebalances; a
    static signal has turnover near 0, a daily-flipper sits near 0.5.
    """
    df = pd.DataFrame(
        {
            "instrument_id": panel.frame["instrument_id"].to_numpy(),
            "date": panel.frame["date"].to_numpy(),
            "feature": feature.to_numpy(),
        }
    )
    df["rank"] = df.groupby("date", sort=False)["feature"].transform(
        lambda s: s.rank(method="average", pct=True)
    )
    df = df.sort_values(["instrument_id", "date"])
    df["prev_rank"] = df.groupby("instrument_id", sort=False)["rank"].shift(1)
    df["abs_change"] = (df["rank"] - df["prev_rank"]).abs()
    per_date = df.groupby("date", sort=True)["abs_change"].mean()
    return per_date


def _pooled_correlation(a: pd.Series, b: pd.Series) -> float:
    """Pooled Pearson correlation of two Series after dropping NaN."""
    df = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy()}).dropna()
    if len(df) < 2:
        return float("nan")
    return float(df["a"].corr(df["b"]))


# ---------------------------------------------------------------------------
# Top-level evidence computation
# ---------------------------------------------------------------------------


def compute_evidence(
    expression: Expression,
    panel: MarketPanel,
    labels: pd.Series,
    *,
    baseline_features: Mapping[str, pd.Series] | None = None,
    cache: ExpressionCache | None = None,
) -> CandidateEvidence:
    """Score ``expression`` as a candidate alpha.

    Parameters
    ----------
    expression:
        The AST to evaluate.
    panel:
        Validated :class:`MarketPanel`.
    labels:
        Forward-return Series, index-aligned to ``panel.frame``. Use
        :func:`make_forward_return_labels` to derive from the panel.
    baseline_features:
        Optional ``{name: Series}`` of already-admitted alpha values
        (or any baseline features) to check correlation against. The
        max absolute pooled correlation feeds the admission gate's
        diversity check.
    cache:
        Optional :class:`ExpressionCache` so a mining loop can share
        cached sub-expressions across candidates.

    Returns
    -------
    CandidateEvidence
    """
    cache = cache if cache is not None else ExpressionCache()
    feature = evaluate_expression(panel, expression, cache=cache)
    feature = feature.replace([np.inf, -np.inf], np.nan)

    coverage = int((feature.notna() & labels.notna()).sum())
    pearson_per_date = _per_date_correlation(panel, feature, labels, method="pearson")
    spearman_per_date = _per_date_correlation(panel, feature, labels, method="spearman")

    mean_ic = float(pearson_per_date.mean()) if pearson_per_date.notna().any() else float("nan")
    rank_ic = float(spearman_per_date.mean()) if spearman_per_date.notna().any() else float("nan")

    if pearson_per_date.notna().sum() >= 2:
        ic_std = float(pearson_per_date.std(ddof=1))
        icir = mean_ic / ic_std if ic_std > 0 else float("nan")
    else:
        icir = float("nan")

    turnover_series = _per_date_turnover(panel, feature)
    turnover = float(turnover_series.mean()) if turnover_series.notna().any() else float("nan")

    if baseline_features:
        correlations = [
            abs(_pooled_correlation(feature, baseline)) for baseline in baseline_features.values()
        ]
        finite = [c for c in correlations if np.isfinite(c)]
        corr_max = max(finite) if finite else float("nan")
    else:
        corr_max = float("nan")

    return CandidateEvidence(
        mean_ic=mean_ic,
        rank_ic=rank_ic,
        icir=icir,
        turnover=turnover,
        coverage=coverage,
        correlation_to_baseline_max=corr_max,
        n_dates=int(pearson_per_date.notna().sum()),
    )


__all__ = [
    "CandidateEvidence",
    "compute_evidence",
    "make_forward_return_labels",
]
