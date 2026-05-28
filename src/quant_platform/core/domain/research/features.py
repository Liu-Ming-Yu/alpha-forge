"""Feature-governance and point-in-time feature domain models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from quant_platform.core.domain.signals.feature_inputs import (
    FeatureInputContext,
    FeatureRequestContext,
    coerce_feature_input_context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


class FeatureProductionState(StrEnum):
    """Governed lifecycle state for one feature definition."""

    DRAFT = "draft"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


class FeatureExpectedSign(StrEnum):
    """Expected relationship between feature value and forward return."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NON_MONOTONIC = "non_monotonic"


@dataclass(frozen=True)
class FeatureDefinition:
    """Human-owned contract for one feature before it enters governed models."""

    name: str
    version: str
    owner: str
    economic_thesis: str
    source_datasets: tuple[str, ...]
    required_lags: tuple[str, ...]
    valid_universe: str
    expected_sign: FeatureExpectedSign
    horizon_days: int
    expected_turnover: str
    state: FeatureProductionState = FeatureProductionState.DRAFT
    failure_modes: tuple[str, ...] = ()
    risk_exposures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "version",
            "owner",
            "economic_thesis",
            "valid_universe",
            "expected_turnover",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")
        if not self.source_datasets:
            raise ValueError("source_datasets must not be empty")
        if self.state == FeatureProductionState.LIVE and not self.failure_modes:
            raise ValueError("live feature definitions must document failure_modes")


@dataclass(frozen=True)
class FeatureAuditResult:
    """Durable summary of the latest feature-level production audit."""

    audit_id: uuid.UUID
    feature_name: str
    feature_version: str
    feature_set_version: str
    as_of: datetime
    sample_start: datetime
    sample_end: datetime
    status: FeatureProductionState
    passed: bool
    metrics: Mapping[str, float]
    gate_results: Mapping[str, bool]
    artifact_uri: str
    schema_hash: str
    code_commit: str
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("feature_name must not be empty")
        if not self.feature_version.strip():
            raise ValueError("feature_version must not be empty")
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        if self.sample_end < self.sample_start:
            raise ValueError("sample_end must be >= sample_start")
        for name, value in (
            ("as_of", self.as_of),
            ("sample_start", self.sample_start),
            ("sample_end", self.sample_end),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.passed and self.blockers:
            raise ValueError("passed feature audits cannot have blockers")


@dataclass(frozen=True)
class FeatureRequest:
    """Typed point-in-time request for one versioned feature set."""

    feature_set_version: str
    instruments: tuple[uuid.UUID, ...]
    start: datetime
    end: datetime
    as_of: datetime
    context: FeatureRequestContext = field(default_factory=FeatureInputContext)
    strategy_run_id: uuid.UUID = field(default_factory=uuid.uuid4)
    artifact_uri: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", coerce_feature_input_context(self.context))
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("FeatureRequest instruments must be unique")
        for name, value in (
            ("start", self.start),
            ("end", self.end),
            ("as_of", self.as_of),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def input_context(self) -> FeatureInputContext:
        """Return the normalised typed feature-input context."""
        if isinstance(self.context, FeatureInputContext):
            return self.context
        raise TypeError("FeatureRequest context was not normalised")


@dataclass(frozen=True)
class FeatureVector:
    """A named, versioned set of numeric features for one instrument at one point in time.

    Args:
        vector_id: Stable content-addressed UUID.
        instrument_id: FK to Instrument.
        as_of: The timestamp at which these features were computed, UTC.
        feature_set_version: Semantic version of the feature computation code
            that produced this vector (e.g. "1.3.0").
        features: Immutable mapping of feature name → float value.
            NaN values are allowed only for explicitly optional features.
        strategy_run_id: The StrategyRun that requested this computation.

    Failure semantics:
        Consumers must check for NaN in required features before using the
        vector for signal scoring.  The signal service must reject vectors
        with required-feature NaN values rather than propagating them.
    """

    vector_id: uuid.UUID
    instrument_id: uuid.UUID
    as_of: datetime
    feature_set_version: str
    features: Mapping[str, float]
    strategy_run_id: uuid.UUID
    artifact_uri: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.available_at is not None and self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")


@dataclass(frozen=True)
class FeatureResult:
    """Typed result for a versioned feature computation."""

    feature_set_version: str
    vectors: tuple[FeatureVector, ...]
    diagnostics: Mapping[str, object]
    passed: bool

    def __post_init__(self) -> None:
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        for vector in self.vectors:
            if vector.feature_set_version != self.feature_set_version:
                raise ValueError("FeatureResult vector feature_set_version mismatch")
        if self.passed and self.diagnostics.get("blockers"):
            raise ValueError("passed FeatureResult cannot contain blockers")


@dataclass(frozen=True)
class FeatureDataset:
    """Immutable feature dataset version used by both research and live."""

    dataset_id: uuid.UUID
    feature_set_version: str
    as_of: datetime
    available_at: datetime
    schema_hash: str
    source_dataset_ids: tuple[uuid.UUID, ...]
    artifact_uri: str
    quality_status: str = "pending"

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.available_at < self.as_of:
            raise ValueError("available_at must be >= as_of")
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        if not self.schema_hash.strip():
            raise ValueError("schema_hash must not be empty")
        if len(set(self.source_dataset_ids)) != len(self.source_dataset_ids):
            raise ValueError("source_dataset_ids must be unique")


@dataclass(frozen=True)
class FeatureSnapshot:
    """Production feature input tied to an approved FeatureDataset."""

    dataset: FeatureDataset
    vectors: tuple[FeatureVector, ...]
    as_of: datetime

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.dataset.quality_status != "approved":
            raise ValueError("FeatureSnapshot requires an approved FeatureDataset")
        seen: set[uuid.UUID] = set()
        for vector in self.vectors:
            if vector.instrument_id in seen:
                raise ValueError("FeatureSnapshot vectors must have unique instruments")
            seen.add(vector.instrument_id)
            if vector.feature_set_version != self.dataset.feature_set_version:
                raise ValueError("vector feature_set_version does not match dataset")
            if vector.as_of > self.as_of:
                raise ValueError("vector.as_of must be <= snapshot as_of")

    def as_feature_data(self) -> dict[uuid.UUID, dict[str, float]]:
        """Return the controller feature-data mapping."""
        return {vector.instrument_id: dict(vector.features) for vector in self.vectors}
