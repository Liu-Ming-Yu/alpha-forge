"""Model artifact and alpha-readiness research domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


class PromotionState(StrEnum):
    """Governed alpha/model deployment state."""

    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


@dataclass(frozen=True)
class ModelArtifact:
    """Immutable model artifact manifest registered before any promotion."""

    artifact_id: uuid.UUID
    model_name: str
    model_version: str
    artifact_uri: str
    artifact_hash: str
    feature_schema_hash: str
    training_start: datetime
    training_end: datetime
    created_at: datetime
    promotion_state: PromotionState = PromotionState.SHADOW
    rollback_artifact_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not self.artifact_hash.strip():
            raise ValueError("artifact_hash must not be empty")
        if not self.feature_schema_hash.strip():
            raise ValueError("feature_schema_hash must not be empty")
        if self.training_end <= self.training_start:
            raise ValueError("training_end must be after training_start")
        for name, value in (
            ("training_start", self.training_start),
            ("training_end", self.training_end),
            ("created_at", self.created_at),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class ModelCard:
    """Human-reviewable model card linked to an immutable artifact."""

    card_id: uuid.UUID
    artifact_id: uuid.UUID
    model_name: str
    model_version: str
    owner: str
    intended_use: str
    metrics: Mapping[str, float]
    risk_notes: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("owner must not be empty")
        if not self.intended_use.strip():
            raise ValueError("intended_use must not be empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class AlphaReadinessReport:
    """Governance report for promoting or retaining an alpha source."""

    report_id: uuid.UUID
    alpha_source: str
    as_of: datetime
    promotion_state: PromotionState
    passed: bool
    metrics: Mapping[str, float]
    drift: Mapping[str, float]
    rollback_target: str = ""

    def __post_init__(self) -> None:
        if not self.alpha_source.strip():
            raise ValueError("alpha_source must not be empty")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
