"""Market bar primitives and vendor bar batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

INTRADAY_BAR_SECONDS = 60
SUPPORTED_BAR_SECONDS = frozenset({INTRADAY_BAR_SECONDS, 300, 900, 3600, 86400})


@dataclass(frozen=True)
class MarketBar:
    """A single OHLCV price bar for one instrument at one resolution.

    Args:
        bar_id: Stable content-addressed UUID (deterministic from
            instrument_id + timestamp + bar_seconds).
        instrument_id: FK to Instrument.instrument_id.
        timestamp: Bar open time, timezone-aware UTC.
        bar_seconds: Bar duration in seconds (e.g. 86400 = daily).
        open: Opening price, split/dividend adjusted.
        high: High price, split/dividend adjusted.
        low: Low price, split/dividend adjusted.
        close: Closing price, split/dividend adjusted.
        volume: Shares traded during the bar.
        vwap: Volume-weighted average price; None if not provided by source.
        is_complete: False if the bar is still open (streaming intraday case).

    Failure semantics:
        Raises ValueError on construction if price invariants are violated.
        The data service must reject and log bars that fail validation rather
        than storing them silently.
    """

    bar_id: uuid.UUID
    instrument_id: uuid.UUID
    timestamp: datetime
    bar_seconds: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None
    is_complete: bool = True

    def __post_init__(self) -> None:
        if self.bar_seconds not in SUPPORTED_BAR_SECONDS:
            raise ValueError(f"bar_seconds {self.bar_seconds} not in {SUPPORTED_BAR_SECONDS}")
        if not (self.low <= self.open <= self.high):
            raise ValueError("price invariant violated: low <= open <= high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("price invariant violated: low <= close <= high")
        if self.low <= 0:
            raise ValueError("low must be > 0")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True)
class VendorBarBatch:
    """Canonical bars returned by one historical-data vendor adapter.

    ``source_uri`` should identify the immutable upstream artifact or request
    trace that produced the batch.  Promotion gates use ``fetched_at`` and the
    coverage fields to distinguish a reproducible vendor dataset from an ad
    hoc fetch.
    """

    vendor: str
    source_uri: str
    fetched_at: datetime
    bar_seconds: int
    bars: tuple[MarketBar, ...]
    coverage: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vendor.strip():
            raise ValueError("vendor must not be empty")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")
        if self.fetched_at.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        if self.bar_seconds not in SUPPORTED_BAR_SECONDS:
            raise ValueError(f"bar_seconds {self.bar_seconds} not in {SUPPORTED_BAR_SECONDS}")
        for bar in self.bars:
            if bar.bar_seconds != self.bar_seconds:
                raise ValueError("all bars in a VendorBarBatch must share bar_seconds")
