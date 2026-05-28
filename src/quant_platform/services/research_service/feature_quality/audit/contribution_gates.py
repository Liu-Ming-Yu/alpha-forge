"""Economic, cost, and contribution feature-audit gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    FeatureDefinition,
    FeatureExpectedSign,
    FeatureProductionState,
)
from quant_platform.services.research_service.feature_quality.audit.calculations import (
    baseline_score_by_day,
    combine_scores,
    daily_score_ic,
    feature_score_by_day,
    feature_weighted_returns,
    max_baseline_corr,
    sign_multiplier,
)
from quant_platform.services.research_service.reports.statistics import mean

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.feature_quality.audit.thresholds import (
        FeatureAuditThresholds,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def evaluate_economic_gate(
    feature: FeatureDefinition,
    stability: Mapping[str, object],
) -> dict[str, object]:
    raw_metrics = stability.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("stability.metrics must be a mapping")
    metrics = {str(key): value for key, value in raw_metrics.items()}
    mean_ic = float(metrics.get("ic_mean", 0.0))
    thesis_length = len(feature.economic_thesis.strip())
    sign_ok = feature.expected_sign == FeatureExpectedSign.NON_MONOTONIC or mean_ic >= 0.0
    blockers: list[str] = []
    if thesis_length < 30:
        blockers.append("economic_thesis is too short for production review")
    if not feature.required_lags:
        blockers.append("required_lags must document point-in-time lag assumptions")
    if not feature.failure_modes and feature.state in {
        FeatureProductionState.PAPER,
        FeatureProductionState.LIVE,
    }:
        blockers.append("paper/live feature must document failure_modes")
    if not sign_ok:
        blockers.append("observed IC sign conflicts with expected_sign")
    return {
        "passed": not blockers,
        "metrics": {
            "economic_thesis_length": float(thesis_length),
            "economic_sign_ok": 1.0 if sign_ok else 0.0,
            "economic_failure_modes": float(len(feature.failure_modes)),
            "economic_risk_exposures": float(len(feature.risk_exposures)),
        },
        "blockers": blockers,
    }


def evaluate_cost_gate(
    feature: FeatureDefinition,
    rows: Sequence[SupervisedAlphaSample],
    thresholds: FeatureAuditThresholds,
    slippage_bps: float,
) -> dict[str, object]:
    daily_returns, turnover = feature_weighted_returns(
        rows,
        feature.name,
        sign=sign_multiplier(feature.expected_sign),
        slippage_bps=slippage_bps,
    )
    gross_returns, _ = feature_weighted_returns(
        rows,
        feature.name,
        sign=sign_multiplier(feature.expected_sign),
        slippage_bps=0.0,
    )
    avg_turnover = mean(turnover)
    net_mean = mean(daily_returns)
    gross_mean = mean(gross_returns)
    metrics = {
        "cost_avg_turnover": avg_turnover,
        "cost_gross_mean_return": gross_mean,
        "cost_net_mean_return": net_mean,
        "cost_drag": gross_mean - net_mean,
        "cost_slippage_bps_per_turnover": slippage_bps,
    }
    blockers: list[str] = []
    if avg_turnover > thresholds.max_turnover:
        blockers.append(f"avg turnover {avg_turnover:.4f} > {thresholds.max_turnover:.4f}")
    if net_mean <= thresholds.min_net_mean_return:
        blockers.append(f"net mean return {net_mean:.6f} <= {thresholds.min_net_mean_return:.6f}")
    return {"passed": not blockers, "metrics": metrics, "blockers": blockers}


def evaluate_incremental_gate(
    feature: FeatureDefinition,
    samples: Sequence[SupervisedAlphaSample],
    thresholds: FeatureAuditThresholds,
    baseline_features: Sequence[str],
) -> dict[str, object]:
    candidate = feature_score_by_day(samples, feature.name, sign_multiplier(feature.expected_sign))
    candidate_ic = mean(daily_score_ic(candidate))
    baseline = baseline_score_by_day(samples, baseline_features)
    baseline_ic = mean(daily_score_ic(baseline)) if baseline else 0.0
    combined = combine_scores(candidate, baseline) if baseline else candidate
    combined_ic = mean(daily_score_ic(combined))
    delta_ic = combined_ic - baseline_ic if baseline else candidate_ic
    max_corr = max_baseline_corr(samples, feature.name, baseline_features)
    metrics = {
        "incremental_candidate_ic": candidate_ic,
        "incremental_baseline_ic": baseline_ic,
        "incremental_combined_ic": combined_ic,
        "incremental_delta_ic": delta_ic,
        "incremental_max_baseline_correlation": max_corr,
        "incremental_baseline_feature_count": float(len(baseline_features)),
    }
    blockers: list[str] = []
    if delta_ic < thresholds.min_incremental_ic_delta:
        blockers.append(f"delta IC {delta_ic:.4f} < {thresholds.min_incremental_ic_delta:.4f}")
    if baseline_features and max_corr > thresholds.max_baseline_correlation:
        blockers.append(
            f"max baseline correlation {max_corr:.4f} > {thresholds.max_baseline_correlation:.4f}"
        )
    return {"passed": not blockers, "metrics": metrics, "blockers": blockers}


__all__ = ["evaluate_cost_gate", "evaluate_economic_gate", "evaluate_incremental_gate"]
