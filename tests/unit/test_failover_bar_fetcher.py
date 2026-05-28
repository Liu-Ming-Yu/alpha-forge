"""Unit tests for FailoverBarFetcher."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.feeds.failover_bar_fetcher import (
    FailoverBarFetcher,
)


def _inst(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _bar(
    iid: uuid.UUID,
    day: date,
) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=iid,
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        bar_seconds=86400,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=1,
    )


@pytest.mark.asyncio
async def test_merge_fills_instruments_primary_missed() -> None:
    a = _inst("A")
    b = _inst("B")
    start = date(2026, 1, 1)
    end = date(2026, 1, 10)

    async def primary(instrs: list[Instrument], s: date, e: date) -> list[MarketBar]:
        assert len(instrs) == 2
        return [_bar(a.instrument_id, end)]

    async def secondary(instrs: list[Instrument], s: date, e: date) -> list[MarketBar]:
        assert {x.symbol for x in instrs} == {"B"}
        return [_bar(b.instrument_id, end)]

    fb = FailoverBarFetcher(primary, secondaries=[secondary])
    out = await fb([a, b], start, end)
    by_sym = {x.instrument_id: x for x in out}
    assert a.instrument_id in by_sym
    assert b.instrument_id in by_sym


@pytest.mark.asyncio
async def test_primary_exception_falls_back_to_all_secondary() -> None:
    a = _inst()
    start = end = date(2026, 1, 2)

    async def primary(_instrs: list[Instrument], _s: date, _e: date) -> list[MarketBar]:
        raise RuntimeError("ib down")

    async def secondary(instrs: list[Instrument], _s: date, e: date) -> list[MarketBar]:
        assert len(instrs) == 1
        return [_bar(a.instrument_id, e)]

    fb = FailoverBarFetcher(primary, secondaries=[secondary])
    out = await fb([a], start, end)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_secondary_fill_raises_keeps_primary_rows() -> None:
    a = _inst("A")
    b = _inst("B")
    start = date(2026, 1, 1)
    end = date(2026, 1, 10)

    async def primary(_instrs: list[Instrument], _s: date, e: date) -> list[MarketBar]:
        return [_bar(a.instrument_id, e)]

    async def secondary(_instrs: list[Instrument], _s: date, _e: date) -> list[MarketBar]:
        raise RuntimeError("tiingo down")

    fb = FailoverBarFetcher(primary, secondaries=[secondary])
    out = await fb([a, b], start, end)
    assert len(out) == 1
    assert out[0].instrument_id == a.instrument_id


@pytest.mark.asyncio
async def test_all_primary_happy_no_secondary_call() -> None:
    a = _inst()
    start = end = date(2026, 1, 2)
    called = False

    async def primary(_instrs: list[Instrument], _s: date, e: date) -> list[MarketBar]:
        return [_bar(a.instrument_id, e)]

    async def secondary(_instrs: list[Instrument], _s: date, _e: date) -> list[MarketBar]:
        nonlocal called
        called = True
        return []

    fb = FailoverBarFetcher(primary, secondaries=[secondary])
    out = await fb([a], start, end)
    assert len(out) == 1
    assert not called
