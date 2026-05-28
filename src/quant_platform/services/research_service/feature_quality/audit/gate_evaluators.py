"""Pure evaluators for institutional feature-audit gates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_platform.services.research_service.feature_quality.audit.calculations import (
    availability_violations,
    daily_ic,
    lagged_feature_ic,
    permuted_ic,
    sign_multiplier,
    values_by_day,
)
from quant_platform.services.research_service.feature_quality.audit.contribution_gates import (
    evaluate_cost_gate,
    evaluate_economic_gate,
    evaluate_incremental_gate,
)
from quant_platform.services.research_service.reports.statistics import (
    bootstrap_mean_ci,
    lag1_autocorr,
    mean,
    negative_streak,
    rolling_mean,
    std_sample,
    winsor_impact,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.core.domain.research import FeatureDefinition
    from quant_platform.services.research_service.feature_quality.audit.thresholds import (
        FeatureAuditThresholds,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def evaluate_noise_gate(
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    thresholds: FeatureAuditThresholds,
) -> dict[str, object]:
    values = [float(row.features[feature_name]) for row in samples if feature_name in row.features]
    total = len(samples)
    finite = [value for value in values if math.isfinite(value)]
    coverage = len(finite) / max(1, total)
    grouped_values = values_by_day(samples, feature_name)
    dispersion_by_day = [std_sample(vals) for vals in grouped_values.values() if len(vals) >= 2]
    mean_dispersion = mean(dispersion_by_day)
    unique_ratio = len({round(value, 12) for value in finite}) / max(1, len(finite))
    daily_unique_ratios = [
        _unique_ratio(vals)
        for vals in grouped_values.values()
        if len([value for value in vals if math.isfinite(value)]) >= 2
    ]
    mean_daily_unique_ratio = mean(daily_unique_ratios)
    rank_normalized = _looks_rank_normalized(
        finite=finite,
        mean_daily_unique_ratio=mean_daily_unique_ratio,
        thresholds=thresholds,
    )
    zero_fraction = sum(1 for value in finite if abs(value) < 1e-12) / max(1, len(finite))
    winsorized_impact = winsor_impact(finite)
    autocorr = lag1_autocorr([mean(vals) for _, vals in sorted(grouped_values.items())])
    metrics = {
        "noise_coverage": coverage,
        "noise_mean_cross_sectional_dispersion": mean_dispersion,
        "noise_unique_ratio": unique_ratio,
        "noise_mean_daily_unique_ratio": mean_daily_unique_ratio,
        "noise_rank_normalized": 1.0 if rank_normalized else 0.0,
        "noise_zero_fraction": zero_fraction,
        "noise_winsor_impact": winsorized_impact,
        "noise_lag1_autocorr": autocorr,
    }
    blockers: list[str] = []
    if coverage < thresholds.min_coverage:
        blockers.append(f"coverage {coverage:.3f} < {thresholds.min_coverage:.3f}")
    if mean_dispersion < thresholds.min_dispersion:
        blockers.append(f"dispersion {mean_dispersion:.4f} < {thresholds.min_dispersion:.4f}")
    if rank_normalized:
        if mean_daily_unique_ratio < thresholds.min_unique_ratio:
            blockers.append(
                "daily_unique_ratio "
                f"{mean_daily_unique_ratio:.3f} < {thresholds.min_unique_ratio:.3f}"
            )
    elif unique_ratio < thresholds.min_unique_ratio:
        blockers.append(f"unique_ratio {unique_ratio:.3f} < {thresholds.min_unique_ratio:.3f}")
    if zero_fraction > thresholds.max_zero_fraction:
        blockers.append(f"zero_fraction {zero_fraction:.3f} > {thresholds.max_zero_fraction:.3f}")
    if thresholds.require_rank_normalized and not rank_normalized:
        out_of_range = bool(finite) and (min(finite) < -1.000001 or max(finite) > 1.000001)
        detail = (
            "values fall outside [-1, 1]" if out_of_range else "insufficient cross-sectional spread"
        )
        blockers.append(
            f"feature is not cross-sectionally rank-normalized ({detail}); "
            "alpha features must be winsorized and rank-normalized to [-1, 1]"
        )
    return {"passed": not blockers, "metrics": metrics, "blockers": blockers}


def _unique_ratio(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return len({round(value, 12) for value in finite}) / max(1, len(finite))


def _looks_rank_normalized(
    *,
    finite: Sequence[float],
    mean_daily_unique_ratio: float,
    thresholds: FeatureAuditThresholds,
) -> bool:
    if not finite:
        return False
    if min(finite) < -1.000001 or max(finite) > 1.000001:
        return False
    if max(finite) - min(finite) < thresholds.min_dispersion:
        return False
    return mean_daily_unique_ratio >= thresholds.min_unique_ratio


def evaluate_leakage_gate(
    feature: FeatureDefinition,
    samples: Sequence[SupervisedAlphaSample],
    rows: Sequence[SupervisedAlphaSample],
    thresholds: FeatureAuditThresholds,
    rng_seed: int,
) -> dict[str, object]:
    signed_ic = daily_ic(rows, feature.name, sign=sign_multiplier(feature.expected_sign))
    mean_ic = mean(signed_ic)
    lagged_ic = lagged_feature_ic(rows, feature.name, sign=sign_multiplier(feature.expected_sign))
    shuffled_ic = permuted_ic(rows, feature.name, rng_seed)
    timestamp_violations = availability_violations(samples)
    groups = len({row.as_of.date().isoformat() for row in rows})
    metrics = {
        "leakage_abs_mean_ic": abs(mean_ic),
        "leakage_lagged_ic": lagged_ic,
        "leakage_permuted_ic": shuffled_ic,
        "leakage_timestamp_violations": float(timestamp_violations),
        "leakage_daily_groups": float(groups),
        "leakage_horizon_days": float(feature.horizon_days),
    }
    blockers: list[str] = []
    if groups < thresholds.min_daily_groups:
        blockers.append(f"daily_groups {groups} < {thresholds.min_daily_groups}")
    if timestamp_violations:
        blockers.append(f"{timestamp_violations} samples violate available_at/source availability")
    if abs(mean_ic) > thresholds.max_abs_ic:
        blockers.append(f"abs mean IC {abs(mean_ic):.4f} > leakage cap {thresholds.max_abs_ic:.4f}")
    if (
        abs(mean_ic) >= thresholds.min_oos_ic
        and abs(lagged_ic) < abs(mean_ic) * thresholds.min_lag_ic_fraction
    ):
        blockers.append("lagged IC collapses below required fraction of same-day IC")
    if abs(shuffled_ic) >= abs(mean_ic) and abs(mean_ic) >= thresholds.min_oos_ic:
        blockers.append("permuted control is as strong as candidate feature")
    return {"passed": not blockers, "metrics": metrics, "blockers": blockers}


def evaluate_stability_gate(
    feature: FeatureDefinition,
    rows: Sequence[SupervisedAlphaSample],
    thresholds: FeatureAuditThresholds,
    rng_seed: int,
) -> dict[str, object]:
    signed = sign_multiplier(feature.expected_sign)
    daily_values = daily_ic(rows, feature.name, sign=signed)
    mean_ic = mean(daily_values)
    ic_std = std_sample(daily_values)
    icir = mean_ic / ic_std if ic_std > 0 else (999.0 if mean_ic > 0 else 0.0)
    ci_low, ci_high = bootstrap_mean_ci(daily_values, seed=rng_seed)
    neg_streak = negative_streak(daily_values)
    metrics = {
        "ic_mean": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "ic_rolling_20": rolling_mean(daily_values, 20),
        "ic_rolling_60": rolling_mean(daily_values, 60),
        "ic_rolling_120": rolling_mean(daily_values, 120),
        "ic_bootstrap_p05": ci_low,
        "ic_bootstrap_p95": ci_high,
        "ic_negative_streak": float(neg_streak),
        "ic_observations": float(len(daily_values)),
    }
    blockers: list[str] = []
    if mean_ic < thresholds.min_oos_ic:
        blockers.append(f"mean IC {mean_ic:.4f} < {thresholds.min_oos_ic:.4f}")
    if icir < thresholds.min_icir:
        blockers.append(f"ICIR {icir:.4f} < {thresholds.min_icir:.4f}")
    if neg_streak > thresholds.max_negative_ic_streak:
        blockers.append(f"negative IC streak {neg_streak} > {thresholds.max_negative_ic_streak}")
    return {"passed": not blockers, "metrics": metrics, "blockers": blockers}


__all__ = [
    "evaluate_cost_gate",
    "evaluate_economic_gate",
    "evaluate_incremental_gate",
    "evaluate_leakage_gate",
    "evaluate_noise_gate",
    "evaluate_stability_gate",
]
