"""Campaign eligibility gates for research promotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.services.research_service.sampling.factory_models import (
        AlphaEligibilityThresholds,
    )


def eligibility(
    metrics: Mapping[str, float],
    thresholds: AlphaEligibilityThresholds,
) -> dict[str, object]:
    streak_passed, streak_threshold = _streak_check(metrics, thresholds)
    checks = [
        (
            "oos_rolling_ic",
            metrics["oos_rolling_ic"] > thresholds.min_oos_rolling_ic,
            metrics["oos_rolling_ic"],
            thresholds.min_oos_rolling_ic,
        ),
        (
            "ic_60d",
            metrics["ic_60d"] > thresholds.min_ic_60d,
            metrics["ic_60d"],
            thresholds.min_ic_60d,
        ),
        (
            # Fold-level streak — see AlphaEligibilityThresholds docstring for
            # why daily-IC streaks are not used as the eligibility gate, and for
            # the drawdown-conditioned relaxation (ADR-004 Option D).
            "fold_negative_ic_streak",
            streak_passed,
            metrics["fold_negative_ic_streak"],
            streak_threshold,
        ),
        (
            "max_drawdown",
            metrics["max_drawdown"] > thresholds.max_drawdown,
            metrics["max_drawdown"],
            thresholds.max_drawdown,
        ),
        (
            "slippage_adjusted_sharpe",
            metrics["slippage_adjusted_sharpe"] >= thresholds.min_slippage_adjusted_sharpe,
            metrics["slippage_adjusted_sharpe"],
            thresholds.min_slippage_adjusted_sharpe,
        ),
    ]
    if thresholds.min_bootstrap_ic_p05 is not None:
        # Robustness gate (ADR-004 v3): the IC must be statistically positive —
        # the 5th percentile of the block-bootstrapped fold-IC distribution above
        # the bound. Replaces the brittle, OOS-unstable negative-IC-streak count.
        checks.append(
            (
                "bootstrap_ic_p05",
                metrics["bootstrap_ic_p05"] > thresholds.min_bootstrap_ic_p05,
                metrics["bootstrap_ic_p05"],
                thresholds.min_bootstrap_ic_p05,
            )
        )
    payload = [
        {"name": name, "passed": passed, "actual": actual, "threshold": threshold}
        for name, passed, actual, threshold in checks
    ]
    return {"passed": all(item["passed"] for item in payload), "checks": payload}


def _streak_check(
    metrics: Mapping[str, float],
    thresholds: AlphaEligibilityThresholds,
) -> tuple[bool, int]:
    """Evaluate the fold-negative-IC-streak gate.

    Returns ``(passed, effective_threshold)``. The effective threshold is the
    largest streak this gate tolerated for these metrics, so the audit row
    self-documents which bound applied.

    Without a configured relaxation
    (``max_fold_negative_ic_streak_if_dd_contained is None``) this is the legacy
    behaviour exactly — ``streak <= max_fold_negative_ic_streak`` — and the
    drawdown-during-streak metric is never read, so callers that predate the
    field are unaffected.

    With a relaxation configured (ADR-004 Option D), a streak above the strict
    floor is tolerated up to the relaxed cap **only if** the drawdown during the
    worst streak stayed inside the candidate drawdown bound
    (``max_drawdown``). If the construction failed to contain that episode, the
    strict floor applies and the relaxation is forfeit — "we trust the
    construction iff it actually protected you," enforced per-episode.
    """
    streak = metrics["fold_negative_ic_streak"]
    floor = thresholds.max_fold_negative_ic_streak
    if streak <= floor:
        # Within the strict floor — no relaxation needed, so the
        # drawdown-during-streak metric is never required here.
        return True, floor
    relaxed_cap = thresholds.max_fold_negative_ic_streak_if_dd_contained
    if relaxed_cap is None:
        return False, floor
    containment_bound = thresholds.streak_containment_max_drawdown
    if containment_bound is None:
        containment_bound = thresholds.max_drawdown
    dd_contained = metrics["max_drawdown_during_worst_streak"] >= containment_bound
    effective_threshold = relaxed_cap if dd_contained else floor
    return streak <= effective_threshold, effective_threshold


__all__ = ["eligibility"]
