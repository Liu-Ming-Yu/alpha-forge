"""Promotion gate for auto-discovered formulaic alphas.

Three things matter when deciding whether to admit a candidate:

1. **Stand-alone strength.** The candidate's IC, ICIR, and turnover
   have to clear absolute thresholds.
2. **Coverage.** The candidate has to actually produce values on
   enough of the panel — a lookback that eats the dataset isn't a
   usable alpha.
3. **Diversity.** The candidate has to be distinct from already-
   admitted alphas. Correlation pruning rejects "yet another rank
   of close" candidates.

:class:`AdmissionThresholds` ships the numerical knobs. The defaults
match the brief's "minimum coverage, positive OOS IC, acceptable
turnover, acceptable feature correlation" guidance without being
artificially tight — production runs should tune them against the
target universe's IC distribution before trusting the output.

:class:`AdmissionGate.evaluate` takes one candidate's provenance and
the list of already-admitted candidates, returns an
:class:`AdmissionDecision` with a stable, human-readable reason
string. The reason ends up on
:attr:`AutoAlphaProvenance.admission_reason` so the audit trail is
self-explaining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.research.features.formulaic.mining.provenance import (
        AutoAlphaProvenance,
    )


@dataclass(frozen=True)
class AdmissionThresholds:
    """Numeric thresholds the gate enforces.

    Attributes
    ----------
    min_rank_ic:
        Floor on the mean per-date Spearman correlation between the
        candidate and the forward-return label. Rank IC is more
        outlier-robust than Pearson IC, which is why it carries the
        primary signal threshold.
    min_icir:
        Floor on Information-Coefficient Information-Ratio — the
        consistency of the signal across dates. A candidate with high
        mean IC but high IC volatility (i.e. low ICIR) is admitting a
        regime risk, not a stable alpha.
    max_turnover:
        Ceiling on per-date L1 rank turnover. Realistic alphas turn
        over slowly enough that transaction costs don't eat the
        signal; values past ~0.4 typically can't survive cost
        adjustment in a US equities cash-account regime.
    min_coverage_ratio:
        Floor on ``evidence.coverage / panel_size``. A candidate that
        only produces values on 10% of the panel is reporting a
        misleading IC computed over a curated subset.
    max_correlation_to_admitted:
        Ceiling on the maximum absolute correlation against any
        already-admitted alpha. New candidates have to add diversity,
        not redundancy. The brief warns about "obvious sector / size
        beta contamination"; the correlation check here catches it
        as a side effect when the baselines include sector / size
        proxies.
    min_n_dates:
        Floor on the number of distinct dates that contributed to
        ``mean_ic``. Stops the gate from admitting a "great IC" that
        actually has 3 dates of data behind it.
    """

    min_rank_ic: float = 0.02
    min_icir: float = 0.1
    max_turnover: float = 0.4
    min_coverage_ratio: float = 0.5
    max_correlation_to_admitted: float = 0.7
    min_n_dates: int = 30
    # Walk-forward-only thresholds — ignored when the evidence is a
    # single-pass :class:`CandidateEvidence`. When the evidence is a
    # :class:`WalkForwardEvidence`, both are enforced.
    max_fold_negative_ic_streak: int = 2
    min_n_folds_valid: int = 3

    def __post_init__(self) -> None:
        if self.max_fold_negative_ic_streak < 0:
            raise ValueError("max_fold_negative_ic_streak must be >= 0")
        if self.min_n_folds_valid < 1:
            raise ValueError("min_n_folds_valid must be >= 1")
        if not (-1.0 <= self.min_rank_ic <= 1.0):
            raise ValueError("min_rank_ic must lie in [-1, 1]")
        if self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be > 0")
        if not (0.0 <= self.min_coverage_ratio <= 1.0):
            raise ValueError("min_coverage_ratio must lie in [0, 1]")
        if not (0.0 <= self.max_correlation_to_admitted <= 1.0):
            raise ValueError("max_correlation_to_admitted must lie in [0, 1]")
        if self.min_n_dates < 1:
            raise ValueError("min_n_dates must be >= 1")


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of one gate evaluation.

    Attributes
    ----------
    admitted:
        ``True`` if every threshold passed.
    reason:
        Human-readable rationale. For an admission this is
        ``"admitted"``; for rejections it names the failing rule
        ("rank_ic below threshold", "correlation_to_admitted above
        threshold", …) and the offending values.
    failed_checks:
        Tuple of check names that failed. Useful for batch analysis
        of which threshold is bottlenecking a given search run.
    """

    admitted: bool
    reason: str
    failed_checks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AdmissionGate:
    """Stateless gate that scores a candidate against thresholds."""

    thresholds: AdmissionThresholds = field(default_factory=AdmissionThresholds)

    def evaluate(
        self,
        candidate: AutoAlphaProvenance,
        already_admitted: Sequence[AutoAlphaProvenance],
        *,
        panel_size: int,
    ) -> AdmissionDecision:
        """Return an :class:`AdmissionDecision` for ``candidate``.

        Parameters
        ----------
        candidate:
            The provenance record with populated :attr:`evidence`.
        already_admitted:
            Provenance records the gate has admitted earlier in this
            run. Used only to look up the candidate's
            ``correlation_to_baseline_max`` against — the caller is
            responsible for passing the same baseline list the
            evidence was scored against.
        panel_size:
            Number of rows in the panel the evidence was computed on
            (typically ``len(panel.frame)``). Used to convert the
            ``coverage`` count into a ratio for the threshold.
        """
        del already_admitted  # consulted indirectly via evidence.correlation_to_baseline_max
        evidence = candidate.evidence
        failed: list[str] = []
        reasons: list[str] = []

        def _fail(name: str, message: str) -> None:
            failed.append(name)
            reasons.append(message)

        if _is_nan_or_lt(evidence.rank_ic, self.thresholds.min_rank_ic):
            _fail(
                "rank_ic",
                f"rank_ic={evidence.rank_ic:.4f} < min_rank_ic={self.thresholds.min_rank_ic}",
            )
        if _is_nan_or_lt(evidence.icir, self.thresholds.min_icir):
            _fail(
                "icir",
                f"icir={evidence.icir:.4f} < min_icir={self.thresholds.min_icir}",
            )
        if _is_nan_or_gt(evidence.turnover, self.thresholds.max_turnover):
            _fail(
                "turnover",
                f"turnover={evidence.turnover:.4f} > max_turnover={self.thresholds.max_turnover}",
            )
        if panel_size > 0:
            coverage_ratio = evidence.coverage / panel_size
            if coverage_ratio < self.thresholds.min_coverage_ratio:
                _fail(
                    "coverage_ratio",
                    f"coverage_ratio={coverage_ratio:.3f} < "
                    f"min_coverage_ratio={self.thresholds.min_coverage_ratio}",
                )
        if evidence.n_dates < self.thresholds.min_n_dates:
            _fail(
                "n_dates",
                f"n_dates={evidence.n_dates} < min_n_dates={self.thresholds.min_n_dates}",
            )
        # NaN correlation means "no baseline supplied" — that's fine,
        # not a rejection. Only fail when the value is finite AND too high.
        if not math.isnan(evidence.correlation_to_baseline_max) and (
            evidence.correlation_to_baseline_max > self.thresholds.max_correlation_to_admitted
        ):
            _fail(
                "correlation_to_admitted",
                f"|corr|={evidence.correlation_to_baseline_max:.3f} > "
                f"max_correlation_to_admitted={self.thresholds.max_correlation_to_admitted}",
            )

        # Walk-forward-only checks. Duck-typed: the WF evidence
        # carries ``fold_negative_ic_streak`` and ``n_folds_valid``;
        # the single-pass evidence doesn't, so ``getattr(..., None)``
        # silently skips the check.
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

        if not failed:
            return AdmissionDecision(admitted=True, reason="admitted", failed_checks=())
        return AdmissionDecision(
            admitted=False,
            reason="; ".join(reasons),
            failed_checks=tuple(failed),
        )


def _is_nan_or_lt(value: float, threshold: float) -> bool:
    """``True`` if ``value`` is NaN OR strictly less than ``threshold``."""
    return math.isnan(value) or value < threshold


def _is_nan_or_gt(value: float, threshold: float) -> bool:
    """``True`` if ``value`` is NaN OR strictly greater than ``threshold``."""
    return math.isnan(value) or value > threshold


__all__ = [
    "AdmissionDecision",
    "AdmissionGate",
    "AdmissionThresholds",
]
