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
            # why daily-IC streaks are not used as the eligibility gate.
            "fold_negative_ic_streak",
            metrics["fold_negative_ic_streak"] <= thresholds.max_fold_negative_ic_streak,
            metrics["fold_negative_ic_streak"],
            thresholds.max_fold_negative_ic_streak,
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
    payload = [
        {"name": name, "passed": passed, "actual": actual, "threshold": threshold}
        for name, passed, actual, threshold in checks
    ]
    return {"passed": all(item["passed"] for item in payload), "checks": payload}


__all__ = ["eligibility"]
