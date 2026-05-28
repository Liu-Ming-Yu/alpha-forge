"""Prediction evidence domain models for governed alpha promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid


@dataclass(frozen=True)
class PredictionResult:
    """One point-in-time forecast produced by an alpha source."""

    prediction_id: uuid.UUID
    strategy_run_id: uuid.UUID
    instrument_id: uuid.UUID
    source: str
    model_version: str
    as_of: datetime
    horizon: str
    expected_return: float
    rank_score: float
    confidence: float
    feature_schema_hash: str
    calibration_bucket: str
    blockers: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("PredictionResult.as_of must be timezone-aware")
        if not self.source.strip():
            raise ValueError("PredictionResult.source must be non-empty")
        if not self.model_version.strip():
            raise ValueError("PredictionResult.model_version must be non-empty")
        if not self.horizon.strip():
            raise ValueError("PredictionResult.horizon must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("PredictionResult.confidence must be in [0, 1]")
        if not self.feature_schema_hash.strip():
            raise ValueError("PredictionResult.feature_schema_hash must be non-empty")
        if not self.calibration_bucket.strip():
            raise ValueError("PredictionResult.calibration_bucket must be non-empty")


@dataclass(frozen=True)
class ForecastEvidence:
    """Aggregated forecast evidence for one promoted source."""

    source: str
    model_version: str
    as_of: datetime
    horizon: str
    observations: int
    mean_confidence: float
    latest_prediction_at: datetime | None
    stale_after: timedelta
    blockers: tuple[str, ...] = ()
    calibration_buckets: tuple[str, ...] = ()
    feature_schema_hashes: tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        if self.latest_prediction_at is None:
            return True
        return (
            self.latest_prediction_at.astimezone(UTC)
            < self.as_of.astimezone(UTC) - self.stale_after
        )

    @property
    def passed(self) -> bool:
        return (
            self.observations > 0
            and not self.stale
            and not self.blockers
            and 0.0 <= self.mean_confidence <= 1.0
        )
