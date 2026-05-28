"""Market-data dataset and quorum evidence domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data.bars import SUPPORTED_BAR_SECONDS

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class DataLakeLayer(StrEnum):
    """Industrial data-lake layer for market data lineage."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DataQualityStatus(StrEnum):
    """Dataset quality state used by live fail-closed gates."""

    PENDING = "pending"
    APPROVED = "approved"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class BarDataset:
    """Versioned bar dataset consumed by research and live feature pipelines."""

    dataset_id: uuid.UUID
    layer: DataLakeLayer
    vendor: str
    bar_seconds: int
    start_at: datetime
    end_at: datetime
    as_of: datetime
    available_at: datetime
    schema_hash: str
    source_uri: str
    row_count: int
    quality: DataQualityStatus = DataQualityStatus.PENDING

    def __post_init__(self) -> None:
        if self.bar_seconds not in SUPPORTED_BAR_SECONDS:
            raise ValueError(f"bar_seconds {self.bar_seconds} not in {SUPPORTED_BAR_SECONDS}")
        if self.end_at < self.start_at:
            raise ValueError("end_at must be >= start_at")
        if self.available_at < self.as_of:
            raise ValueError("available_at must be >= as_of")
        for name, value in (
            ("start_at", self.start_at),
            ("end_at", self.end_at),
            ("as_of", self.as_of),
            ("available_at", self.available_at),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if not self.vendor.strip():
            raise ValueError("vendor must not be empty")
        if not self.schema_hash.strip():
            raise ValueError("schema_hash must not be empty")
        if self.row_count < 0:
            raise ValueError("row_count must be >= 0")


@dataclass(frozen=True)
class DatasetQuorumEvidence:
    """Evidence that independent vendors agree enough for live use."""

    evidence_id: uuid.UUID
    dataset_kind: str
    as_of: datetime
    vendors: tuple[str, ...]
    passed: bool
    required_vendor_count: int = 2
    max_disagreement_bps: Decimal = Decimal("50")
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.dataset_kind.strip():
            raise ValueError("dataset_kind must not be empty")
        cleaned = tuple(vendor.strip() for vendor in self.vendors if vendor.strip())
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("vendors must be unique")
        object.__setattr__(self, "vendors", cleaned)
        if self.required_vendor_count < 1:
            raise ValueError("required_vendor_count must be >= 1")
        if self.max_disagreement_bps < Decimal("0"):
            raise ValueError("max_disagreement_bps must be >= 0")
        if self.passed and len(cleaned) < self.required_vendor_count:
            raise ValueError("passed quorum requires enough vendors")
