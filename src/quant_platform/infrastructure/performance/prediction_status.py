"""Pure prediction-evidence status calculations for performance repositories."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import ForecastEvidence, PredictionResult

if TYPE_CHECKING:
    from datetime import datetime


def build_forecast_evidence(
    source: str,
    *,
    model_version: str | None,
    as_of: datetime,
    rows: list[PredictionResult],
    stale_after_hours: int,
    min_confidence: float,
) -> ForecastEvidence:
    """Summarise prediction freshness and quality for one source."""
    filtered = [
        row
        for row in rows
        if row.source == source and (model_version is None or row.model_version == model_version)
    ]
    blockers: list[str] = []
    if not filtered:
        blockers.append("no prediction evidence recorded")

    row_blockers = sorted({blocker for row in filtered for blocker in row.blockers})
    blockers.extend(row_blockers)

    latest = max((row.as_of for row in filtered), default=None)
    mean_confidence = sum(row.confidence for row in filtered) / len(filtered) if filtered else 0.0
    if filtered and mean_confidence < min_confidence:
        blockers.append(f"mean confidence {mean_confidence:.4f} below minimum {min_confidence:.4f}")

    model = model_version or (filtered[-1].model_version if filtered else "")
    horizon = filtered[-1].horizon if filtered else ""
    buckets = tuple(sorted({row.calibration_bucket for row in filtered}))
    schema_hashes = tuple(sorted({row.feature_schema_hash for row in filtered}))
    return ForecastEvidence(
        source=source,
        model_version=model,
        as_of=as_of,
        horizon=horizon,
        observations=len(filtered),
        mean_confidence=mean_confidence,
        latest_prediction_at=latest,
        stale_after=timedelta(hours=max(1, stale_after_hours)),
        blockers=tuple(blockers),
        calibration_buckets=buckets,
        feature_schema_hashes=schema_hashes,
    )
