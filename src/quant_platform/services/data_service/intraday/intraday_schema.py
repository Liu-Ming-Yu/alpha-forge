"""Canonical intraday bar schema identity."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from quant_platform.services.data_service.intraday.intraday_validation import (
    INTRADAY_BAR_SECONDS,
    ensure_utc,
)

if TYPE_CHECKING:
    from datetime import datetime

INTRADAY_SCHEMA_COLUMNS = (
    "instrument_id|symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap?",
    "is_complete?",
)
INTRADAY_SCHEMA_HASH = hashlib.sha256("|".join(INTRADAY_SCHEMA_COLUMNS).encode("utf-8")).hexdigest()


def canonical_intraday_bar_id(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    bar_seconds: int = INTRADAY_BAR_SECONDS,
) -> uuid.UUID:
    """Return deterministic ID for one canonical bar timestamp."""
    ts = ensure_utc(timestamp).isoformat()
    return uuid.uuid5(uuid.NAMESPACE_URL, f"bar:{instrument_id}:{ts}:{bar_seconds}")


__all__ = [
    "INTRADAY_SCHEMA_COLUMNS",
    "INTRADAY_SCHEMA_HASH",
    "canonical_intraday_bar_id",
]
