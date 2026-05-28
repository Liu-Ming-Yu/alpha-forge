"""Prediction evidence serialization and artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.core.domain.production import ForecastEvidence, PredictionResult


def prediction_result_payload(result: PredictionResult) -> dict[str, object]:
    """Return JSON-safe payload for one prediction result."""
    payload = asdict(result)
    payload["prediction_id"] = str(result.prediction_id)
    payload["strategy_run_id"] = str(result.strategy_run_id)
    payload["instrument_id"] = str(result.instrument_id)
    payload["as_of"] = result.as_of.isoformat()
    payload["blockers"] = list(result.blockers)
    return payload


def forecast_evidence_payload(evidence: ForecastEvidence) -> dict[str, object]:
    """Return JSON-safe payload for aggregate forecast evidence."""
    return {
        "source": evidence.source,
        "model_version": evidence.model_version,
        "as_of": evidence.as_of.isoformat(),
        "horizon": evidence.horizon,
        "observations": evidence.observations,
        "mean_confidence": evidence.mean_confidence,
        "latest_prediction_at": (
            evidence.latest_prediction_at.isoformat()
            if evidence.latest_prediction_at is not None
            else None
        ),
        "stale_after_seconds": evidence.stale_after.total_seconds(),
        "stale": evidence.stale,
        "passed": evidence.passed,
        "blockers": list(evidence.blockers),
        "calibration_buckets": list(evidence.calibration_buckets),
        "feature_schema_hashes": list(evidence.feature_schema_hashes),
    }


def write_prediction_artifact(
    *,
    results: list[PredictionResult],
    evidence: ForecastEvidence,
    output: Path,
) -> Path:
    """Write prediction evidence as a machine-readable artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "evidence": forecast_evidence_payload(evidence),
                "predictions": [prediction_result_payload(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output
