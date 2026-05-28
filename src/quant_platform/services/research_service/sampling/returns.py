"""Forward-return and bar-history helpers for supervised samples."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.services.research_service.sampling.dates import _ensure_utc

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from quant_platform.core.contracts import HistoricalDataStore
    from quant_platform.core.domain.market_data import MarketBar


async def _bar_history_by_instrument(
    *,
    bar_store: HistoricalDataStore,
    instrument_ids: Sequence[uuid.UUID],
    as_of_dates: Sequence[datetime],
    horizon_days: int,
    bar_seconds: int,
) -> dict[uuid.UUID, list[MarketBar]]:
    if not as_of_dates:
        return {}
    start = min(as_of_dates) - timedelta(days=7)
    end = max(as_of_dates) + timedelta(days=horizon_days + 14)
    history: dict[uuid.UUID, list[MarketBar]] = {}
    for instrument_id in instrument_ids:
        history[instrument_id] = sorted(
            await bar_store.get_bars(instrument_id, bar_seconds, start, end),
            key=lambda bar: bar.timestamp,
        )
    return history


async def _forward_return(
    *,
    bar_store: HistoricalDataStore,
    instrument_id: uuid.UUID,
    as_of: datetime,
    horizon_days: int,
    bar_seconds: int,
) -> float | None:
    target = as_of + timedelta(days=horizon_days)
    bars = await bar_store.get_bars(
        instrument_id,
        bar_seconds,
        as_of - timedelta(days=7),
        target + timedelta(days=14),
    )
    if not bars:
        return None
    return _forward_return_from_bars(
        bars=sorted(bars, key=lambda bar: bar.timestamp),
        as_of=as_of,
        horizon_days=horizon_days,
    )


def _forward_return_from_bars(
    *,
    bars: Sequence[MarketBar],
    as_of: datetime,
    horizon_days: int,
) -> float | None:
    if not bars:
        return None
    target = as_of + timedelta(days=horizon_days)
    ordered = [
        bar
        for bar in bars
        if as_of - timedelta(days=7) <= _ensure_utc(bar.timestamp) <= target + timedelta(days=14)
    ]
    entry_candidates = [bar for bar in ordered if _ensure_utc(bar.timestamp) <= as_of]
    exit_candidates = [bar for bar in ordered if _ensure_utc(bar.timestamp) >= target]
    if not entry_candidates or not exit_candidates:
        return None
    entry = Decimal(entry_candidates[-1].close)
    exit_ = Decimal(exit_candidates[0].close)
    if entry <= 0:
        return None
    # Log return: symmetric and better-conditioned for cross-sectional ranking
    # than simple return, which is distorted by large outliers.
    return float(math.log(float(exit_) / float(entry)))


__all__ = ["_bar_history_by_instrument", "_forward_return", "_forward_return_from_bars"]
