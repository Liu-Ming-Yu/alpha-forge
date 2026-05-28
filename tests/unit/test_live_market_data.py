"""Tests for PollingMarketDataProvider staleness detection (P0-3)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.services.data_service.feeds.live_market_data import PollingMarketDataProvider


def _bar(*, minutes_old: float, bar_seconds: int = 60) -> object:
    """Return a minimal MarketBar-like object with a given age."""
    from quant_platform.core.domain.market_data import MarketBar

    ts = datetime.now(tz=UTC) - timedelta(minutes=minutes_old)
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        timestamp=ts,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=1000,
        bar_seconds=bar_seconds,
        is_complete=True,
    )


@pytest.mark.asyncio
async def test_stale_bar_raises_data_staleness_error() -> None:
    """A cached bar older than max_bar_age_minutes must raise DataStalenessError."""
    instrument_id = uuid.uuid4()
    stale_bar = _bar(minutes_old=6)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None  # fetch fails, fall through to cache

    provider = PollingMarketDataProvider(_fetch, max_bar_age_minutes=5)
    # Seed the cache manually
    provider._cache[(instrument_id, 60)] = stale_bar  # type: ignore[attr-defined]

    with pytest.raises(DataStalenessError) as exc_info:
        await provider.get_last_bar(instrument_id, 60)

    assert exc_info.value.instrument_id == instrument_id
    assert exc_info.value.max_age_minutes == 5


@pytest.mark.asyncio
async def test_fresh_bar_returned_normally() -> None:
    """A cached bar within max_bar_age_minutes must be returned without error."""
    instrument_id = uuid.uuid4()
    fresh_bar = _bar(minutes_old=4)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None

    provider = PollingMarketDataProvider(_fetch, max_bar_age_minutes=5)
    provider._cache[(instrument_id, 60)] = fresh_bar  # type: ignore[attr-defined]

    result = await provider.get_last_bar(instrument_id, 60)
    assert result is fresh_bar


@pytest.mark.asyncio
async def test_no_max_age_returns_stale_silently() -> None:
    """When max_bar_age_minutes is None (default), stale bars are returned without error."""
    instrument_id = uuid.uuid4()
    ancient_bar = _bar(minutes_old=1440)  # 24 hours old

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None

    provider = PollingMarketDataProvider(_fetch, max_bar_age_minutes=None)
    provider._cache[(instrument_id, 60)] = ancient_bar  # type: ignore[attr-defined]

    result = await provider.get_last_bar(instrument_id, 60)
    assert result is ancient_bar


@pytest.mark.asyncio
async def test_successful_fetch_updates_cache_and_bypasses_staleness() -> None:
    """A successful live fetch always returns the fresh bar, even if cache is stale."""
    instrument_id = uuid.uuid4()
    fresh_bar = _bar(minutes_old=0)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return fresh_bar

    provider = PollingMarketDataProvider(_fetch, max_bar_age_minutes=1)
    # Seed cache with stale bar; the fresh fetch should win.
    provider._cache[(instrument_id, 60)] = _bar(minutes_old=99)  # type: ignore[attr-defined]

    result = await provider.get_last_bar(instrument_id, 60)
    assert result is fresh_bar


@pytest.mark.asyncio
async def test_daily_bar_uses_daily_staleness_window() -> None:
    """Daily bars can be older than the intraday max while still usable."""
    instrument_id = uuid.uuid4()
    daily_bar = _bar(minutes_old=1173, bar_seconds=86400)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None

    provider = PollingMarketDataProvider(
        _fetch,
        max_bar_age_minutes=60,
        daily_max_bar_age_minutes=2880,
    )
    provider._cache[(instrument_id, 86400)] = daily_bar  # type: ignore[attr-defined]

    result = await provider.get_last_bar(instrument_id, 86400)
    assert result is daily_bar


@pytest.mark.asyncio
async def test_daily_bar_beyond_daily_window_raises() -> None:
    instrument_id = uuid.uuid4()
    daily_bar = _bar(minutes_old=3000, bar_seconds=86400)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None

    provider = PollingMarketDataProvider(
        _fetch,
        max_bar_age_minutes=60,
        daily_max_bar_age_minutes=2880,
    )
    provider._cache[(instrument_id, 86400)] = daily_bar  # type: ignore[attr-defined]

    with pytest.raises(DataStalenessError) as exc_info:
        await provider.get_last_bar(instrument_id, 86400)

    assert exc_info.value.max_age_minutes == 2880


@pytest.mark.asyncio
async def test_zero_max_age_disables_relevant_staleness_check() -> None:
    instrument_id = uuid.uuid4()
    stale_bar = _bar(minutes_old=3000)

    async def _fetch(_id: uuid.UUID, _seconds: int):
        return None

    provider = PollingMarketDataProvider(_fetch, max_bar_age_minutes=0)
    provider._cache[(instrument_id, 60)] = stale_bar  # type: ignore[attr-defined]

    result = await provider.get_last_bar(instrument_id, 60)
    assert result is stale_bar
