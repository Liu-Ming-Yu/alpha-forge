"""Point-in-time security-master domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import date, datetime

    from quant_platform.core.domain.instruments.core import Instrument


class SecurityMasterQuality(StrEnum):
    """Point-in-time security-master record quality state."""

    PENDING = "pending"
    APPROVED = "approved"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class SecurityMasterRecord:
    """Point-in-time security-master row for one instrument.

    This is the production source of truth for canonical identifiers and live
    tradability metadata.  Broker-specific identifiers are stored as data in
    ``identifiers`` so the domain model remains broker-agnostic.
    """

    record_id: uuid.UUID
    instrument: Instrument
    as_of: datetime
    available_at: datetime
    identifiers: Mapping[str, str] = field(default_factory=dict)
    primary_exchange: str = ""
    country: str = "US"
    source: str = "security_master"
    quality: SecurityMasterQuality = SecurityMasterQuality.APPROVED

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.available_at < self.as_of:
            raise ValueError("available_at must be >= as_of")
        if self.instrument.active and self.quality != SecurityMasterQuality.APPROVED:
            raise ValueError("active instruments require approved security-master metadata")


@dataclass(frozen=True)
class SymbolHistory:
    """Point-in-time ticker history for an instrument."""

    history_id: uuid.UUID
    instrument_id: uuid.UUID
    symbol: str
    valid_from: date
    valid_to: date | None = None
    source: str = "security_master"

    def __post_init__(self) -> None:
        if not self.symbol.isupper():
            raise ValueError("symbol must be upper-case")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")
