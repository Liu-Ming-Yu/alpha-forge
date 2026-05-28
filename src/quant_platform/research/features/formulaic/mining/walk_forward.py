"""Walk-forward (K-fold OOS) evidence for alpha mining.

Single-pass IC scoring (the :mod:`.evidence` path) computes one IC
over the whole panel — convenient for fast search loops, but it has
selection bias: every alpha that beats the gate was *evaluated* on
the same dates that *admitted* it. Walk-forward evidence fixes that
by splitting the panel's date range into K contiguous folds and
reporting per-fold ICs plus their aggregate. A candidate that scores
well on one fold but collapses on another lights up via a low
``oos_icir`` and a long ``fold_negative_ic_streak`` even when its
pooled IC looks fine.

Mining WF vs campaign WF
------------------------

The campaign evaluator (``services/research_service/campaigns/
evaluation/walk_forward.py``) does a richer thing — it fits feature
weights per fold and rolls portfolio diagnostics — because the
candidate there is a linear blend, not a fixed expression. In
formulaic mining the alpha is *the* model; nothing is fit. So the
mining WF reduces to:

1. Partition the panel's unique dates into K contiguous chunks.
2. For each chunk, compute the per-date IC inside that chunk.
3. Aggregate to mean-IC + IC-IR + worst negative streak.

That's enough to flag regime-fragile alphas without dragging in the
full campaign WF surface.

Embargo
-------

Forward-return labels at row ``date=d`` look forward ``horizon``
trading days, so a label whose ``d`` is near a fold boundary reaches
into the next fold's territory. ``MiningFoldConfig.embargo_days``
drops rows whose date is within that many days of a fold boundary so
fold-level IC distributions stay distinct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.formulaic.mining.evidence import (
    _per_date_correlation,
    _per_date_turnover,
    _pooled_correlation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.panel import MarketPanel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiningFoldConfig:
    """Settings for K-fold OOS evaluation.

    Attributes
    ----------
    n_folds:
        Number of contiguous date chunks. Practical sweet spot is 4–6
        for a typical 2–3-year panel — fewer folds means more dates
        per fold but less regime-stability signal; more folds means
        each fold's IC is noisier.
    embargo_days:
        Trading-day buffer dropped at each fold boundary. Set to the
        same value as the forward-return horizon so labels that span
        a boundary don't double-count.
    min_test_days:
        A fold is considered "valid" only if it carries at least
        this many distinct dates with non-null
        (feature, label) pairs. Folds below this floor contribute
        ``NaN`` to the per-fold IC sequence and are excluded from the
        ``oos_*`` aggregates.
    """

    n_folds: int = 4
    embargo_days: int = 5
    min_test_days: int = 20

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("MiningFoldConfig.n_folds must be >= 2")
        if self.embargo_days < 0:
            raise ValueError("MiningFoldConfig.embargo_days must be >= 0")
        if self.min_test_days < 1:
            raise ValueError("MiningFoldConfig.min_test_days must be >= 1")


@dataclass(frozen=True)
class WalkForwardEvidence:
    """K-fold OOS evidence for one candidate.

    Field-name compatibility note: the fields that overlap with
    :class:`CandidateEvidence` are spelled identically (``rank_ic``,
    ``icir``, ``turnover``, ``coverage``,
    ``correlation_to_baseline_max``, ``n_dates``) so the existing
    :class:`AdmissionGate` reads them unchanged. The
    ``rank_ic`` / ``icir`` fields carry the **OOS** quantities — the
    fold-level mean and the fold-level IR — not the pooled values.

    Attributes
    ----------
    mean_ic, rank_ic:
        Mean across folds of each fold's per-date IC. ``rank_ic`` is
        the Spearman-correlation aggregate.
    icir:
        ``mean_ic / std(per_fold_mean_ic)``. The OOS IR — high when
        the alpha is consistently profitable across folds, low when
        it's regime-sensitive.
    fold_ics, fold_rank_ics:
        Per-fold mean Pearson / Spearman IC. Length is ``n_folds``;
        folds that fail the ``min_test_days`` check are ``NaN``.
    fold_negative_ic_streak:
        Longest consecutive run of folds with mean IC strictly less
        than zero. Aliased on the gate as
        ``fold_negative_ic_streak`` per the brief's Priority 1
        rename.
    n_folds_valid:
        Number of folds that cleared ``min_test_days``. Provenance
        and the gate both look at this — an alpha "validated" on 1
        fold is no validation at all.
    n_dates:
        Number of distinct dates summed across valid folds.
    turnover, coverage, correlation_to_baseline_max:
        Computed over the full panel (not per-fold). Turnover is
        ~regime-invariant so a single pooled number is informative;
        coverage and correlation are accounting quantities the gate
        consumes the same way as in single-pass mode.
    """

    mean_ic: float
    rank_ic: float
    icir: float
    fold_ics: tuple[float, ...]
    fold_rank_ics: tuple[float, ...]
    fold_negative_ic_streak: int
    n_folds_valid: int
    n_dates: int
    turnover: float
    coverage: int
    correlation_to_baseline_max: float


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def _generate_fold_boundaries(
    sorted_dates: np.ndarray,
    n_folds: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``sorted_dates`` into ``n_folds`` contiguous chunks.

    Returns a list of ``(start, end_inclusive)`` pairs. The split is
    by date-index, not by calendar-time — so ``n_folds=4`` always
    produces 4 chunks of roughly equal length even on irregular date
    indices.
    """
    n = len(sorted_dates)
    # ``np.linspace(0, n, n_folds + 1, dtype=int)`` gives one more
    # boundary than there are folds. Convert each chunk to a
    # (first, last) pair.
    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(n_folds):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if hi <= lo:
            # Fewer than one date per fold; record a degenerate
            # (start, start - 1) pair so the per-fold loop produces
            # an empty mask and ``n_folds_valid`` reflects the
            # collapse.
            pairs.append((pd.Timestamp(sorted_dates[lo - 1]), pd.Timestamp(sorted_dates[lo - 1])))
            continue
        pairs.append((pd.Timestamp(sorted_dates[lo]), pd.Timestamp(sorted_dates[hi - 1])))
    return pairs


def _longest_negative_streak(values: list[float]) -> int:
    """Longest run of consecutive strictly-negative entries in ``values``.

    NaN entries break the run (they are treated as "neither
    negative nor positive"). Used to compute
    :attr:`WalkForwardEvidence.fold_negative_ic_streak`.
    """
    best = 0
    current = 0
    for v in values:
        if v < 0:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_walk_forward_evidence(
    expression: Expression,
    panel: MarketPanel,
    labels: pd.Series,
    *,
    fold_config: MiningFoldConfig,
    baseline_features: Mapping[str, pd.Series] | None = None,
    cache: ExpressionCache | None = None,
) -> WalkForwardEvidence:
    """Score ``expression`` with K-fold OOS evidence.

    See module docstring for the high-level shape. The function takes
    the same arguments as :func:`..evidence.compute_evidence` plus a
    :class:`MiningFoldConfig` controlling the split.
    """
    cache = cache if cache is not None else ExpressionCache()
    feature = evaluate_expression(panel, expression, cache=cache)
    feature = feature.replace([np.inf, -np.inf], np.nan)

    coverage = int((feature.notna() & labels.notna()).sum())

    unique_dates = np.sort(panel.frame["date"].unique())
    if len(unique_dates) < fold_config.n_folds * fold_config.min_test_days:
        # Not enough dates for the requested split. Return all-NaN
        # per-fold values and let the gate reject on n_folds_valid.
        empty: tuple[float, ...] = tuple(float("nan") for _ in range(fold_config.n_folds))
        return WalkForwardEvidence(
            mean_ic=float("nan"),
            rank_ic=float("nan"),
            icir=float("nan"),
            fold_ics=empty,
            fold_rank_ics=empty,
            fold_negative_ic_streak=0,
            n_folds_valid=0,
            n_dates=int((feature.notna() & labels.notna()).sum()),
            turnover=_pooled_turnover(panel, feature),
            coverage=coverage,
            correlation_to_baseline_max=_max_baseline_correlation(feature, baseline_features),
        )

    boundaries = _generate_fold_boundaries(unique_dates, fold_config.n_folds)
    date_array = panel.frame["date"].to_numpy()

    fold_ics: list[float] = []
    fold_rank_ics: list[float] = []
    n_dates_total = 0
    n_valid_folds = 0

    # Embargo is applied symmetrically around fold boundaries: we drop
    # rows within ``embargo_days`` of either end of each fold. The
    # ``business-day`` count is approximated by trading-day-index
    # offset on ``unique_dates``.
    embargo = fold_config.embargo_days

    for fold_index, (start, end) in enumerate(boundaries):
        # Embargo around boundaries other than the panel ends. The
        # first fold's leading edge and the last fold's trailing edge
        # are panel ends — no embargo needed there.
        is_first = fold_index == 0
        is_last = fold_index == fold_config.n_folds - 1

        start_idx = int(np.searchsorted(unique_dates, np.datetime64(start)))
        end_idx = int(np.searchsorted(unique_dates, np.datetime64(end))) + 1

        eff_start_idx = start_idx + (0 if is_first else embargo)
        eff_end_idx = end_idx - (0 if is_last else embargo)

        if eff_end_idx - eff_start_idx < fold_config.min_test_days:
            fold_ics.append(float("nan"))
            fold_rank_ics.append(float("nan"))
            continue

        fold_first = unique_dates[eff_start_idx]
        fold_last = unique_dates[min(eff_end_idx - 1, len(unique_dates) - 1)]
        mask = (date_array >= fold_first) & (date_array <= fold_last)
        if int(mask.sum()) == 0:
            fold_ics.append(float("nan"))
            fold_rank_ics.append(float("nan"))
            continue

        fold_panel_frame = panel.frame.loc[mask].reset_index(drop=True)
        fold_feature = feature.loc[mask].reset_index(drop=True)
        fold_labels = labels.loc[mask].reset_index(drop=True)
        # Wrap into a lightweight panel-shaped object the IC helpers
        # consume. ``_per_date_correlation`` only reads
        # ``panel.frame["date"]`` so a duck-typed namespace is enough;
        # the ``cast`` documents intent without a runtime check.
        fold_panel = cast("MarketPanel", _FoldPanel(frame=fold_panel_frame))

        pearson = _per_date_correlation(fold_panel, fold_feature, fold_labels, method="pearson")
        spearman = _per_date_correlation(fold_panel, fold_feature, fold_labels, method="spearman")
        valid_dates = int(pearson.notna().sum())
        if valid_dates < fold_config.min_test_days:
            fold_ics.append(float("nan"))
            fold_rank_ics.append(float("nan"))
            continue
        fold_ics.append(float(pearson.mean()))
        fold_rank_ics.append(float(spearman.mean()))
        n_dates_total += valid_dates
        n_valid_folds += 1

    finite_ics = [x for x in fold_ics if math.isfinite(x)]
    finite_rank_ics = [x for x in fold_rank_ics if math.isfinite(x)]
    mean_ic = float(np.mean(finite_ics)) if finite_ics else float("nan")
    rank_ic = float(np.mean(finite_rank_ics)) if finite_rank_ics else float("nan")
    if len(finite_ics) >= 2:
        ic_std = float(np.std(finite_ics, ddof=1))
        icir = mean_ic / ic_std if ic_std > 0 else float("nan")
    else:
        icir = float("nan")

    fold_negative_streak = _longest_negative_streak(fold_ics)
    turnover = _pooled_turnover(panel, feature)
    correlation_max = _max_baseline_correlation(feature, baseline_features)

    return WalkForwardEvidence(
        mean_ic=mean_ic,
        rank_ic=rank_ic,
        icir=icir,
        fold_ics=tuple(fold_ics),
        fold_rank_ics=tuple(fold_rank_ics),
        fold_negative_ic_streak=fold_negative_streak,
        n_folds_valid=n_valid_folds,
        n_dates=n_dates_total,
        turnover=turnover,
        coverage=coverage,
        correlation_to_baseline_max=correlation_max,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FoldPanel:
    """Duck-typed MarketPanel substitute for IC helpers.

    The ``_per_date_correlation`` helper only needs ``frame["date"]``;
    bundling the sliced frame here lets us reuse the helper without
    constructing a full MarketPanel (which would re-validate columns
    and re-derive ``returns`` / ``dollar_volume``).
    """

    frame: pd.DataFrame


def _pooled_turnover(panel: MarketPanel, feature: pd.Series) -> float:
    series = _per_date_turnover(panel, feature)
    return float(series.mean()) if series.notna().any() else float("nan")


def _max_baseline_correlation(
    feature: pd.Series,
    baseline_features: Mapping[str, pd.Series] | None,
) -> float:
    if not baseline_features:
        return float("nan")
    correlations = [
        abs(_pooled_correlation(feature, baseline)) for baseline in baseline_features.values()
    ]
    finite = [c for c in correlations if np.isfinite(c)]
    return max(finite) if finite else float("nan")


__all__ = [
    "MiningFoldConfig",
    "WalkForwardEvidence",
    "compute_walk_forward_evidence",
]
