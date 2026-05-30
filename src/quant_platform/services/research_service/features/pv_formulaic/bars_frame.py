"""Adapt live ``MarketBar`` payloads to the OHLCV frame the feature math expects.

First, layering-clean increment of the G-live integration (ADR-011). Pure
``services`` → ``core`` conversion (no dependency on the research feature kernel
that later increments port). The output frame matches
``price_volume.features.REQUIRED_INPUT_COLUMNS`` so the ported feature compute
can consume it unchanged:
``(instrument_id, date, open, high, low, close, volume)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.domain.market_data.bars import MarketBar

#: Column contract of the produced frame, in order. Mirrors
#: ``research.features.price_volume.features.REQUIRED_INPUT_COLUMNS``.
OHLCV_FRAME_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _as_naive_day(timestamp: datetime) -> pd.Timestamp:
    """Normalise a (tz-aware UTC) bar timestamp to a tz-naive calendar day.

    Matches the research bars frame, which is tz-naive at day granularity so
    rolling-window features key on the trading date.
    """
    stamp = pd.Timestamp(timestamp)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def market_bars_to_ohlcv_frame(
    bars_by_instrument: Mapping[uuid.UUID, Sequence[MarketBar]],
    *,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Flatten per-instrument ``MarketBar`` sequences into the OHLCV frame.

    Point-in-time safe: bars with ``timestamp`` strictly after ``as_of`` are
    dropped. Duplicate ``(instrument_id, date)`` rows keep the last bar, and rows
    are sorted by ``(instrument_id, date)`` so rolling-window features compute in
    chronological order. ``instrument_id`` is preserved as the ``uuid.UUID`` so
    the downstream feature bundle can key on it directly.

    Returns an empty frame with :data:`OHLCV_FRAME_COLUMNS` when no bars fall
    within the window.
    """
    rows: list[dict[str, object]] = []
    for instrument_id, bars in bars_by_instrument.items():
        for bar in bars:
            if as_of is not None and bar.timestamp > as_of:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "date": _as_naive_day(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(OHLCV_FRAME_COLUMNS))
    frame = pd.DataFrame(rows, columns=list(OHLCV_FRAME_COLUMNS))
    frame = frame.sort_values(["instrument_id", "date"]).drop_duplicates(
        ["instrument_id", "date"], keep="last"
    )
    return frame.reset_index(drop=True)


__all__ = ["OHLCV_FRAME_COLUMNS", "market_bars_to_ohlcv_frame"]
