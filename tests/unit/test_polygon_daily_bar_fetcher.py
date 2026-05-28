"""Unit tests for PolygonDailyBarFetcher (HTTP mocked) and DataCircuitBreaker integration."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import httpx
import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.services.data_service.feeds.polygon_daily_bar_fetcher import (
    PolygonDailyBarFetcher,
)


def _make_inst(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _polygon_payload(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "request_id": "test",
        "status": "OK",
        "results": [
            {
                "t": 1767312000000,  # 2026-01-02 UTC
                "o": 10.0,
                "h": 10.5,
                "l": 9.5,
                "c": 10.1,
                "v": 1000,
            }
        ],
    }


def test_polygon_requires_api_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PolygonDailyBarFetcher("  ")


@pytest.mark.asyncio
async def test_polygon_call_empty_instruments() -> None:
    fetcher = PolygonDailyBarFetcher("key")
    out = await fetcher([], date(2026, 1, 1), date(2026, 1, 5))
    assert out == []


@pytest.mark.asyncio
async def test_polygon_call_end_before_start() -> None:
    fetcher = PolygonDailyBarFetcher("key")
    inst = _make_inst()
    out = await fetcher([inst], date(2026, 1, 5), date(2026, 1, 1))
    assert out == []


@pytest.mark.asyncio
async def test_polygon_fetch_one_parses_ohlcv() -> None:
    inst = _make_inst()
    fetcher = PolygonDailyBarFetcher("testkey")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "AAPL" in str(request.url)
        assert "testkey" in str(request.url)
        return httpx.Response(200, json=_polygon_payload())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert len(out) == 1
    assert out[0].close == Decimal("10.1")
    assert out[0].bar_seconds == 86400
    assert out[0].instrument_id == inst.instrument_id


@pytest.mark.asyncio
async def test_polygon_fetch_one_404_returns_empty() -> None:
    inst = _make_inst("GONE")
    fetcher = PolygonDailyBarFetcher("testkey")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert out == []
    assert (
        fetcher._cb._failures == 0
    )  # 404 = ticker not found, not a circuit failure  # noqa: SLF001


@pytest.mark.asyncio
async def test_polygon_fetch_one_500_records_failure() -> None:
    inst = _make_inst()
    fetcher = PolygonDailyBarFetcher("testkey")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert out == []
    assert fetcher._cb._failures > 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_polygon_fetch_one_success_resets_cb() -> None:
    inst = _make_inst()
    fetcher = PolygonDailyBarFetcher("testkey")
    fetcher._cb.record_failure()  # noqa: SLF001

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_polygon_payload())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert len(out) == 1
    assert fetcher._cb._failures == 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_polygon_circuit_breaker_skips_when_open() -> None:
    inst = _make_inst()
    fetcher = PolygonDailyBarFetcher("testkey")

    for _ in range(5):
        fetcher._cb.record_failure()  # noqa: SLF001
    assert fetcher._cb.is_open()  # noqa: SLF001

    call_count = 0

    async def _fake_fetch_one(client, i, s, e):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return []

    fetcher._fetch_one = _fake_fetch_one  # type: ignore[method-assign]  # noqa: SLF001
    out = await fetcher([inst], date(2026, 1, 1), date(2026, 1, 5))
    assert out == []
    assert call_count == 0


@pytest.mark.asyncio
async def test_polygon_dot_in_symbol_converted_to_dash() -> None:
    inst = _make_inst("BRK.B")
    fetcher = PolygonDailyBarFetcher("testkey")
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"status": "OK", "results": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert seen_urls and "BRK-B" in seen_urls[0]
