"""Failure attribution for governed feature diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.feature_quality.diagnostics.health import (
    build_feature_diagnostic_health,
)
from quant_platform.services.research_service.feature_quality.failures.metadata import (
    diagnostic_recommendation,
    direction_by_feature,
    family_best_features,
    family_rows,
    feature_families,
    feature_family,
    feature_seed,
    recommended_list,
    recommended_sign,
)
from quant_platform.services.research_service.feature_quality.failures.metrics import (
    correlation_clusters,
    correlation_matrix,
    daily_ic_rows,
    data_validation,
    feature_names,
    ic_summary,
    monthly_ic_rows,
    null_baseline,
    worst_negative_streak_window,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def build_feature_failure_attribution(
    *,
    samples_by_horizon: Mapping[int, Sequence[SupervisedAlphaSample]],
    sample_builds_by_horizon: Mapping[int, Mapping[str, object]],
    feature_set_version: str,
    official_horizon_days: int,
    direction_diagnostics: Mapping[str, object],
    family_metadata: Mapping[str, object],
    date_policy: str,
    nested_object_store_present: bool,
    seed: int = 17,
    permutation_count: int = 200,
    correlation_threshold: float = 0.70,
    candidate_feature_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build a non-training attribution report for quarantined feature evidence."""
    official_samples = tuple(samples_by_horizon.get(official_horizon_days, ()))
    names = (
        tuple(sorted({str(name) for name in candidate_feature_names}))
        if candidate_feature_names is not None
        else feature_names(official_samples)
    )
    directions = direction_by_feature(direction_diagnostics)
    families = feature_families(names, family_metadata)
    matrix = correlation_matrix(official_samples, names)
    family_best = family_best_features(
        families,
        samples_by_horizon,
        directions,
        official_horizon_days,
    )
    features = [
        _feature_row(
            feature_name=name,
            feature_set_version=feature_set_version,
            samples_by_horizon=samples_by_horizon,
            official_horizon_days=official_horizon_days,
            direction_row=directions.get(name, {}),
            family=feature_family(name, families),
            family_best=family_best,
            seed=seed,
            permutation_count=permutation_count,
        )
        for name in names
    ]
    return {
        "feature_set_version": feature_set_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "diagnostic_only": True,
        "promotion_artifacts_written": False,
        "official_horizon_days": official_horizon_days,
        "horizons": sorted(samples_by_horizon),
        "permutation_seed": seed,
        "permutation_count": permutation_count,
        "data_validation": data_validation(
            official_samples=official_samples,
            samples_by_horizon=samples_by_horizon,
            sample_builds_by_horizon=sample_builds_by_horizon,
            date_policy=date_policy,
            nested_object_store_present=nested_object_store_present,
        ),
        "direction_diagnostics": {
            "feature_count": direction_diagnostics.get("feature_count", 0),
            "missing_cards": list(
                cast("list[object]", direction_diagnostics.get("missing_cards", []))
            ),
        },
        "families": family_rows(families, features),
        "features": features,
        "correlation": {
            "threshold": correlation_threshold,
            "matrix": matrix,
            "clusters": correlation_clusters(matrix, correlation_threshold),
        },
        "recommendation_labels": [
            "discard",
            "repair_candidate",
            "family_representative_candidate",
            "needs_more_data",
        ],
    }


def _feature_row(
    *,
    feature_name: str,
    feature_set_version: str,
    samples_by_horizon: Mapping[int, Sequence[SupervisedAlphaSample]],
    official_horizon_days: int,
    direction_row: Mapping[str, object],
    family: str,
    family_best: Mapping[str, str],
    seed: int,
    permutation_count: int,
) -> dict[str, object]:
    sign = recommended_sign(direction_row)
    horizons = {
        str(horizon): ic_summary(samples, feature_name, sign)
        for horizon, samples in sorted(samples_by_horizon.items())
    }
    daily_rows = daily_ic_rows(
        samples_by_horizon.get(official_horizon_days, ()),
        feature_name,
        sign,
    )
    null = null_baseline(
        samples_by_horizon.get(official_horizon_days, ()),
        feature_name,
        sign,
        seed=feature_seed(seed, feature_name),
        count=permutation_count,
    )
    official = cast("Mapping[str, object]", horizons[str(official_horizon_days)])
    health = build_feature_diagnostic_health(
        samples=samples_by_horizon.get(official_horizon_days, ()),
        feature_name=feature_name,
        direction_row=direction_row,
        official=official,
        null=null,
    )
    return {
        "feature_name": feature_name,
        "feature_set_version": feature_set_version,
        "family": family,
        "recommended_orientation": str(direction_row.get("recommended_orientation", "positive")),
        "recommended_passed": bool(direction_row.get("recommended_passed", False)),
        "direction_orientations": direction_row.get("orientations", {}),
        "failed_gates": recommended_list(direction_row, "failed_gates"),
        "gate_blockers": recommended_list(direction_row, "blockers"),
        "horizon_comparison": horizons,
        "daily_ic": {
            "observations": len(daily_rows),
            "worst_negative_streak": worst_negative_streak_window(daily_rows),
            "series": daily_rows,
        },
        "monthly_ic": monthly_ic_rows(daily_rows),
        "null_baseline": null,
        "diagnostic_health": health,
        "diagnostic_recommendation": diagnostic_recommendation(
            official=official,
            null=null,
            feature_name=feature_name,
            family=family,
            family_best=family_best,
        ),
    }


__all__ = ["build_feature_failure_attribution"]
