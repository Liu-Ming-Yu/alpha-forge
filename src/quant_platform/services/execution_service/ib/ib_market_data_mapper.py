"""IB historical market-data mapping helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from quant_platform.core.domain.market_data import MarketBar

RawIbBar = tuple[str, float, float, float, float, int]


def bar_size_string(bar_seconds: int) -> str:
    """Map supported bar seconds to ibapi ``barSizeSetting`` values."""
    mapping = {
        60: "1 min",
        300: "5 mins",
        900: "15 mins",
        3600: "1 hour",
        86400: "1 day",
    }
    return mapping[bar_seconds]


def parse_bar_timestamp(raw: str, bar_seconds: int) -> datetime:
    """Parse an ibapi historical-data bar timestamp into UTC."""
    raw = raw.strip()
    if bar_seconds >= 86400:
        return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=UTC)
    parts = raw.split("  ", 1)
    date_part = parts[0]
    time_part = parts[1] if len(parts) > 1 else "00:00:00"
    time_part = time_part.split(" ", 1)[0]
    return datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H:%M:%S").replace(tzinfo=UTC)


def market_bar_from_raw(
    *,
    instrument_id: uuid.UUID,
    bar_seconds: int,
    raw: RawIbBar,
) -> MarketBar:
    """Build a domain MarketBar from one raw ibapi historical bar tuple."""
    timestamp = parse_bar_timestamp(raw[0], bar_seconds)
    open_px, raw_high, raw_low, close_px = (
        Decimal(str(raw[1])),
        Decimal(str(raw[2])),
        Decimal(str(raw[3])),
        Decimal(str(raw[4])),
    )
    volume = max(0, int(raw[5]))
    high_px = max(raw_high, open_px, close_px)
    low_px = min(raw_low, open_px, close_px)
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=timestamp,
        bar_seconds=bar_seconds,
        open=open_px,
        high=high_px,
        low=low_px,
        close=close_px,
        volume=volume,
        is_complete=True,
    )
