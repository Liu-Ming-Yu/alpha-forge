from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from quant_platform.services.research_service.feature_quality.diagnostics.null_qualification import (
    null_qualified_features,
)
from quant_platform.services.research_service.feature_quality.failures.attribution import (
    build_feature_failure_attribution,
)
from quant_platform.services.research_service.feature_quality.failures.report import (
    render_feature_failure_operator_report,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def _samples(horizon: int) -> tuple[SupervisedAlphaSample, ...]:
    rows: list[SupervisedAlphaSample] = []
    for day in range(260):
        as_of = datetime(2025, 1, 2, tzinfo=UTC) + timedelta(days=day)
        for rank in range(3):
            rows.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=uuid.uuid5(uuid.NAMESPACE_URL, f"instrument:{rank}"),
                    features={
                        "alpha": float(rank),
                        "beta": float(rank * 2),
                        "gamma": float(-rank),
                    },
                    forward_return=float(rank) * 0.01 * (horizon / 21.0),
                )
            )
    return tuple(rows)


def _direction() -> dict[str, object]:
    return {
        "feature_count": 3,
        "missing_cards": [],
        "features": [
            {
                "feature_name": name,
                "recommended_orientation": "positive",
                "recommended_passed": False,
                "orientations": {
                    "positive": {
                        "failed_gates": ["ic_stability"],
                        "blockers": ["negative IC streak 8 > 3"],
                        "metrics": {"ic_mean": 1.0 if name != "gamma" else -1.0},
                    },
                    "negative": {
                        "failed_gates": ["economic_logic"],
                        "blockers": ["observed IC sign conflicts with expected_sign"],
                        "metrics": {"ic_mean": -1.0 if name != "gamma" else 1.0},
                    },
                },
            }
            for name in ("alpha", "beta", "gamma")
        ],
    }


def _payload() -> dict[str, object]:
    samples = {horizon: _samples(horizon) for horizon in (5, 10, 21)}
    return build_feature_failure_attribution(
        samples_by_horizon=samples,
        sample_builds_by_horizon={
            horizon: {"samples": len(rows), "skipped_stale_features": 0}
            for horizon, rows in samples.items()
        },
        feature_set_version="ohlcv-paper-v1.1",
        official_horizon_days=21,
        direction_diagnostics=_direction(),
        family_metadata={"families": {"momentum": ["alpha", "beta"], "trend": ["gamma"]}},
        date_policy="nyse-sessions",
        nested_object_store_present=False,
        seed=17,
        permutation_count=20,
        correlation_threshold=0.70,
    )


def test_feature_failure_attribution_builds_deterministic_null_and_horizon_rows() -> None:
    first = _payload()
    second = _payload()

    alpha = next(row for row in first["features"] if row["feature_name"] == "alpha")
    alpha_again = next(row for row in second["features"] if row["feature_name"] == "alpha")

    assert first["diagnostic_only"] is True
    assert first["promotion_artifacts_written"] is False
    assert set(alpha["horizon_comparison"]) == {"5", "10", "21"}
    assert alpha["null_baseline"] == alpha_again["null_baseline"]
    assert alpha["diagnostic_recommendation"] == "family_representative_candidate"
    assert alpha["diagnostic_health"]["nonzero_fraction"] > 0.0
    assert alpha["diagnostic_health"]["null_margin"] == (
        alpha["horizon_comparison"]["21"]["ic_mean"] - alpha["null_baseline"]["null_p95"]
    )


def test_feature_failure_attribution_clusters_correlated_features_and_renders_report() -> None:
    payload = _payload()

    clusters = payload["correlation"]["clusters"]
    assert any({"alpha", "beta"}.issubset(set(row["features"])) for row in clusters)

    report = render_feature_failure_operator_report(payload)

    assert "ohlcv-paper-v1.1 Feature Failure Attribution" in report
    assert "Promotion artifacts written: `False`" in report
    assert "`momentum`: `alpha`" in report
    assert "## Feature Health" in report
    assert "null_margin=" in report


def test_null_qualified_features_blocks_when_too_few_candidates_beat_p95() -> None:
    payload = _payload()
    qualification = null_qualified_features(payload, official_horizon_days=21)

    assert qualification["audited"] == ("alpha", "beta", "gamma")
    assert len(qualification["qualified"]) < 3
    assert "gamma" in qualification["quarantined"]
