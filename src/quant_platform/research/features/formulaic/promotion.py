"""Promotion gate for auto-discovered formulaic alphas.

The mining loop's :class:`AdmissionGate` is a *low* bar — it lets a
candidate through whenever its single-pass evidence beats a few sane
thresholds, so the operator has a list of "interesting" alphas to
look at. **Promotion** to the production family library
(:mod:`.auto_library`) is a *higher* bar:

* Single-pass IC isn't enough; the candidate must have been scored
  with K-fold OOS evidence (:class:`WalkForwardEvidence`).
* OOS rank-IC must clear a higher floor than the admission gate.
* OOS ICIR must be meaningfully positive.
* No long fold-level negative-IC streaks.
* Enough valid folds to call the evaluation OOS.
* Correlation to already-promoted alphas (the durable library set,
  not just this run's admitted set) must stay below the promotion
  ceiling — stricter than admission's intra-run correlation pruning.

The two-tier design matches the brief's "Do not directly trust auto-
generated features" — admission is for *operator review*, promotion
is for *production inclusion*. The promotion CLI
(:mod:`scripts.promote_alphas`) is the human gate between them.

Content-addressed names
-----------------------

:func:`stable_alpha_id` produces a name like ``auto_alpha_3a2f7b9d``
from a sha256 of the canonical-JSON serialisation of the expression.
Two consequences worth knowing:

1. **Re-running mining with a different seed but discovering the same
   expression produces the same name.** Idempotent. No "duplicate
   alphas under different names" failure mode.
2. **AST-equivalent expressions hash identically.** ``a + b`` and
   ``a + b`` always hit the same name; ``a + b`` vs ``b + a`` do
   *not* (no commutativity canonicalisation today — operator builders
   produce a fixed argument order so this isn't a practical issue).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.formulaic.serialization import (
    expression_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.research.features.formulaic.ast import Expression
    from quant_platform.research.features.formulaic.mining.provenance import (
        AutoAlphaProvenance,
    )


# ---------------------------------------------------------------------------
# Stable, content-addressed naming
# ---------------------------------------------------------------------------


def stable_alpha_id(expression: Expression, *, prefix: str = "auto_alpha") -> str:
    """Return a stable content-addressed name for ``expression``.

    The name is ``{prefix}_{sha256(canonical_json)[:12]}``. 12 hex
    chars = 48 bits of entropy, which is plenty for a library that
    will reasonably hold thousands of alphas; collision probability
    is ~2^-24 at 10k alphas.

    Parameters
    ----------
    expression:
        Any :class:`Expression` subclass instance.
    prefix:
        Name prefix. Defaults to ``"auto_alpha"`` to match the
        brief's recommended naming.

    Returns
    -------
    str
        Deterministic name. Identical expressions always hash to the
        same name; cross-run reproducibility is the point.
    """
    payload = expression_to_dict(expression)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


# ---------------------------------------------------------------------------
# Thresholds + decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionThresholds:
    """Threshold values the :class:`PromotionGate` enforces.

    Every default here is **higher** than the corresponding admission
    threshold (:class:`~.mining.admission.AdmissionThresholds`):
    admission is "interesting", promotion is "reliable enough for
    production".

    Attributes
    ----------
    min_oos_rank_ic:
        Minimum out-of-sample rank-IC (the WF evidence's
        ``rank_ic`` field carries the across-fold mean).
    min_oos_icir:
        Minimum out-of-sample IC information ratio.
    max_fold_negative_ic_streak:
        Maximum allowed run of consecutive folds with negative IC.
        0 = "no fold may be negative" is overly strict on noisy
        panels; 1 is the sweet spot.
    min_n_folds_valid:
        Minimum number of folds where the evaluation actually
        completed (the OOS aggregate should not be supported by 1-2
        folds even if their ICs are excellent).
    max_correlation_to_promoted:
        Maximum |correlation| against any already-promoted alpha's
        series. Strictly lower than the admission gate's
        ``max_correlation_to_admitted`` so production stays
        decorrelated even when intra-run admission is permissive.
    min_n_dates:
        Minimum count of distinct dates carrying a non-null
        (feature, label) pair across the WF evaluation.
    max_turnover:
        Same as the admission ceiling — promotion doesn't ease this
        constraint.
    require_walk_forward:
        ``True`` (default) rejects any candidate whose
        :attr:`AutoAlphaProvenance.evidence` is not a
        :class:`WalkForwardEvidence`. The brief's "walk-forward
        validation" requirement gets enforced at promotion time, not
        admission.
    """

    min_oos_rank_ic: float = 0.04
    min_oos_icir: float = 0.3
    max_fold_negative_ic_streak: int = 1
    min_n_folds_valid: int = 4
    max_correlation_to_promoted: float = 0.5
    min_n_dates: int = 100
    max_turnover: float = 0.4
    require_walk_forward: bool = True

    def __post_init__(self) -> None:
        if not (-1.0 <= self.min_oos_rank_ic <= 1.0):
            raise ValueError("min_oos_rank_ic must lie in [-1, 1]")
        if self.min_oos_icir < 0.0:
            raise ValueError("min_oos_icir must be >= 0")
        if self.max_fold_negative_ic_streak < 0:
            raise ValueError("max_fold_negative_ic_streak must be >= 0")
        if self.min_n_folds_valid < 1:
            raise ValueError("min_n_folds_valid must be >= 1")
        if not (0.0 <= self.max_correlation_to_promoted <= 1.0):
            raise ValueError("max_correlation_to_promoted must lie in [0, 1]")
        if self.min_n_dates < 1:
            raise ValueError("min_n_dates must be >= 1")
        if self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be > 0")


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of evaluating one provenance against the gate."""

    promoted: bool
    reason: str
    failed_checks: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionGate:
    """Stateless gate that scores a candidate for promotion."""

    thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)

    def evaluate(
        self,
        candidate: AutoAlphaProvenance,
        *,
        promoted_feature_series: Mapping[str, pd.Series] | None = None,
        candidate_series: pd.Series | None = None,
    ) -> PromotionDecision:
        """Return a :class:`PromotionDecision` for ``candidate``.

        Parameters
        ----------
        candidate:
            Provenance row with populated :attr:`evidence`. To pass
            with the default thresholds, the evidence must be a
            :class:`WalkForwardEvidence` (carries ``fold_ics``,
            ``fold_negative_ic_streak``, ``n_folds_valid``).
        promoted_feature_series:
            Optional ``{name: Series}`` of already-promoted alphas'
            computed series. Used for correlation pruning. When
            ``None``, the correlation check is skipped (no library
            to compare against).
        candidate_series:
            Optional pre-computed series for ``candidate``. Required
            when ``promoted_feature_series`` is supplied — the
            correlation can't be computed without it. Skipping the
            correlation check by passing ``None`` is a safe default
            for the empty-library bootstrap case.
        """
        evidence = candidate.evidence
        failed: list[str] = []
        reasons: list[str] = []

        def _fail(name: str, message: str) -> None:
            failed.append(name)
            reasons.append(message)

        # ---- WF-required check ----
        if self.thresholds.require_walk_forward:
            wf_marker = getattr(evidence, "fold_ics", None)
            if wf_marker is None:
                _fail(
                    "walk_forward_required",
                    "promotion requires WalkForwardEvidence; candidate has single-pass evidence",
                )

        # ---- IC / IR thresholds (read whichever the evidence carries) ----
        rank_ic = getattr(evidence, "rank_ic", float("nan"))
        if _is_nan_or_lt(rank_ic, self.thresholds.min_oos_rank_ic):
            _fail(
                "oos_rank_ic",
                f"rank_ic={rank_ic:.4f} < min_oos_rank_ic={self.thresholds.min_oos_rank_ic}",
            )
        icir = getattr(evidence, "icir", float("nan"))
        if _is_nan_or_lt(icir, self.thresholds.min_oos_icir):
            _fail(
                "oos_icir",
                f"icir={icir:.4f} < min_oos_icir={self.thresholds.min_oos_icir}",
            )

        turnover = getattr(evidence, "turnover", float("nan"))
        if _is_nan_or_gt(turnover, self.thresholds.max_turnover):
            _fail(
                "turnover",
                f"turnover={turnover:.4f} > max_turnover={self.thresholds.max_turnover}",
            )

        n_dates = getattr(evidence, "n_dates", 0)
        if n_dates < self.thresholds.min_n_dates:
            _fail(
                "n_dates",
                f"n_dates={n_dates} < min_n_dates={self.thresholds.min_n_dates}",
            )

        # ---- WF-specific (skipped automatically for single-pass) ----
        fold_streak = getattr(evidence, "fold_negative_ic_streak", None)
        if fold_streak is not None and fold_streak > self.thresholds.max_fold_negative_ic_streak:
            _fail(
                "fold_negative_ic_streak",
                f"fold_negative_ic_streak={fold_streak} > "
                f"max_fold_negative_ic_streak="
                f"{self.thresholds.max_fold_negative_ic_streak}",
            )
        n_folds_valid = getattr(evidence, "n_folds_valid", None)
        if n_folds_valid is not None and n_folds_valid < self.thresholds.min_n_folds_valid:
            _fail(
                "n_folds_valid",
                f"n_folds_valid={n_folds_valid} < "
                f"min_n_folds_valid={self.thresholds.min_n_folds_valid}",
            )

        # ---- Correlation to already-promoted alphas ----
        if promoted_feature_series and candidate_series is not None:
            max_abs_corr = _max_abs_correlation(candidate_series, promoted_feature_series)
            if (
                np.isfinite(max_abs_corr)
                and max_abs_corr > self.thresholds.max_correlation_to_promoted
            ):
                _fail(
                    "correlation_to_promoted",
                    f"|corr|={max_abs_corr:.3f} > "
                    f"max_correlation_to_promoted={self.thresholds.max_correlation_to_promoted}",
                )

        if not failed:
            return PromotionDecision(promoted=True, reason="promoted", failed_checks=())
        return PromotionDecision(
            promoted=False,
            reason="; ".join(reasons),
            failed_checks=tuple(failed),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_nan_or_lt(value: float, threshold: float) -> bool:
    return math.isnan(value) or value < threshold


def _is_nan_or_gt(value: float, threshold: float) -> bool:
    return math.isnan(value) or value > threshold


def _max_abs_correlation(
    candidate: pd.Series,
    promoted_features: Mapping[str, pd.Series],
) -> float:
    """Maximum |Pearson correlation| against any promoted-feature Series.

    Pooled correlation (not per-date). NaN-aware: rows where either
    side is NaN are dropped. Returns ``nan`` if every comparison was
    undefined (e.g. constant-feature edge case).
    """
    correlations: list[float] = []
    for other in promoted_features.values():
        merged = pd.concat([candidate, other], axis=1).dropna()
        if len(merged) < 2:
            continue
        c1, c2 = merged.iloc[:, 0], merged.iloc[:, 1]
        if c1.std() == 0 or c2.std() == 0:
            continue
        correlations.append(abs(float(c1.corr(c2))))
    finite = [c for c in correlations if np.isfinite(c)]
    return max(finite) if finite else float("nan")


def select_promotions(
    *,
    provenances: Sequence[AutoAlphaProvenance],
    gate: PromotionGate,
    promoted_feature_series: Mapping[str, pd.Series] | None = None,
    candidate_series_by_name: Mapping[str, pd.Series] | None = None,
) -> list[tuple[AutoAlphaProvenance, PromotionDecision]]:
    """Score every provenance against the gate; return (prov, decision) pairs.

    Convenience driver for the promotion CLI. Iterates the
    provenances in the order they were supplied; the correlation
    check sees the running set of promoted alphas plus any from this
    run that already cleared the gate, so the gate enforces
    intra-batch decorrelation in addition to library-level
    decorrelation.

    Parameters
    ----------
    provenances:
        Sequence of provenance records, typically loaded from a
        mining run's JSONL.
    gate:
        Configured :class:`PromotionGate`.
    promoted_feature_series:
        Pre-existing promoted library's feature columns (already-
        admitted alphas from prior runs). Optional.
    candidate_series_by_name:
        ``{provenance.name: pd.Series}`` mapping for the candidates
        in this batch. Required when the gate's correlation check
        should be active; mining drivers that don't materialise the
        candidate series can omit this and skip correlation pruning.

    Returns
    -------
    list[tuple[AutoAlphaProvenance, PromotionDecision]]
        One pair per input provenance, preserving order.
    """
    running_baseline: dict[str, pd.Series] = dict(promoted_feature_series or {})
    pairs: list[tuple[AutoAlphaProvenance, PromotionDecision]] = []
    for prov in provenances:
        candidate_series = (
            candidate_series_by_name.get(prov.name)
            if candidate_series_by_name is not None
            else None
        )
        decision = gate.evaluate(
            prov,
            promoted_feature_series=running_baseline if running_baseline else None,
            candidate_series=candidate_series,
        )
        pairs.append((prov, decision))
        if decision.promoted and candidate_series is not None:
            running_baseline[prov.name] = candidate_series
    return pairs


__all__ = [
    "PromotionDecision",
    "PromotionGate",
    "PromotionThresholds",
    "select_promotions",
    "stable_alpha_id",
]
