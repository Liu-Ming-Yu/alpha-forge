"""Live market-data provider adapters.

These adapters satisfy the ``MarketDataProvider`` protocol without forcing a
specific broker implementation. The caller injects fetch callbacks.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

from quant_platform.core.contracts import MarketDataProvider
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.exceptions import DataStalenessError

LastBarFetcher = Callable[[uuid.UUID, int], Awaitable[MarketBar | None]]


class PollingMarketDataProvider(MarketDataProvider):
    """Poll-based MarketDataProvider over an async ``fetch_last_bar`` callback."""

    def __init__(
        self,
        fetch_last_bar: LastBarFetcher,
        *,
        poll_interval_seconds: float = 1.0,
        max_bar_age_minutes: int | None = None,
        daily_max_bar_age_minutes: int | None = None,
    ) -> None:
        self._fetch_last_bar = fetch_last_bar
        self._poll_interval = poll_interval_seconds
        self._cache: dict[tuple[uuid.UUID, int], MarketBar] = {}
        self._max_bar_age_minutes = _normalized_max_age(max_bar_age_minutes)
        self._daily_max_bar_age_minutes = _normalized_max_age(daily_max_bar_age_minutes)

    async def get_last_bar(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None:
        bar = await self._fetch_last_bar(instrument_id, bar_seconds)
        if bar is not None:
            self._cache[(instrument_id, bar_seconds)] = bar
        else:
            bar = self._cache.get((instrument_id, bar_seconds))

        # Unconditional staleness check — applies to fresh fetches too, since
        # the exchange may be closed and the "fresh" bar might be hours old.
        max_age_minutes = self._max_age_minutes_for(bar_seconds)
        if bar is not None and max_age_minutes is not None:
            age_minutes = (datetime.now(tz=UTC) - bar.timestamp).total_seconds() / 60
            if age_minutes > max_age_minutes:
                raise DataStalenessError(
                    f"bar for instrument {instrument_id} is {age_minutes:.1f} min old "
                    f"(max {max_age_minutes} min)",
                    instrument_id=instrument_id,
                    bar_timestamp=bar.timestamp,
                    max_age_minutes=max_age_minutes,
                )
        return bar

    async def subscribe_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> AsyncIterator[MarketBar]:
        last_bar_id: uuid.UUID | None = None
        while True:
            bar = await self.get_last_bar(instrument_id, bar_seconds)
            if bar is not None and bar.is_complete and bar.bar_id != last_bar_id:
                last_bar_id = bar.bar_id
                yield bar
            await asyncio.sleep(self._poll_interval)

    def _max_age_minutes_for(self, bar_seconds: int) -> int | None:
        if bar_seconds >= 86400:
            return self._daily_max_bar_age_minutes
        return self._max_bar_age_minutes


def _normalized_max_age(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value
