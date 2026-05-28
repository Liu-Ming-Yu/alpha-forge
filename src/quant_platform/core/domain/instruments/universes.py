"""Point-in-time universe snapshot domain model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.core.domain.instruments.security_master import SecurityMasterQuality

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable point-in-time universe membership snapshot."""

    snapshot_id: uuid.UUID
    universe_name: str
    as_of: datetime
    available_at: datetime
    instrument_ids: tuple[uuid.UUID, ...]
    source: str = "security_master"
    quality: SecurityMasterQuality = SecurityMasterQuality.APPROVED

    def __post_init__(self) -> None:
        if not self.universe_name.strip():
            raise ValueError("universe_name must not be empty")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.available_at < self.as_of:
            raise ValueError("available_at must be >= as_of")
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("instrument_ids must be unique")
