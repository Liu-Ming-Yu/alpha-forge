"""Feature-backfill result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class FeatureBackfillDay:
    """One daily backfill result row."""

    as_of: datetime
    instruments_requested: int
    instruments_with_history: int
    feature_vectors: int
    skipped_insufficient_history: int
    skipped_existing_vectors: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "instruments_requested": self.instruments_requested,
            "instruments_with_history": self.instruments_with_history,
            "feature_vectors": self.feature_vectors,
            "skipped_insufficient_history": self.skipped_insufficient_history,
            "skipped_existing_vectors": self.skipped_existing_vectors,
        }


@dataclass(frozen=True)
class FeatureBackfillResult:
    """Machine-readable backfill summary."""

    feature_set_version: str
    date_policy: str
    dry_run: bool
    start: datetime
    end: datetime
    days: tuple[FeatureBackfillDay, ...]

    @property
    def vectors_total(self) -> int:
        return sum(day.feature_vectors for day in self.days)

    @property
    def skipped_insufficient_history(self) -> int:
        return sum(day.skipped_insufficient_history for day in self.days)

    @property
    def skipped_existing_vectors(self) -> int:
        return sum(day.skipped_existing_vectors for day in self.days)

    def to_payload(self) -> dict[str, object]:
        vector_key = "vectors_would_store" if self.dry_run else "vectors_stored"
        return {
            "feature_set_version": self.feature_set_version,
            "date_policy": self.date_policy,
            "dry_run": self.dry_run,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days_processed": len(self.days),
            vector_key: self.vectors_total,
            "skipped_insufficient_history": self.skipped_insufficient_history,
            "skipped_existing_vectors": self.skipped_existing_vectors,
            "daily": [day.to_payload() for day in self.days],
        }


__all__ = ["FeatureBackfillDay", "FeatureBackfillResult"]
