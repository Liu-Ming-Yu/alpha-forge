"""Unit tests for TiingoBarFetcher (HTTP mocked via _fetch_one)."""

from __future__ import annotations

import time
import uuid
from datetime import date
from decimal import Decimal

import httpx
import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.infrastructure.support.circuit_breaker import DataCircuitBreaker
from quant_platform.services.data_service.feeds.tiingo_bar_fetcher import TiingoBarFetcher


def _make_inst() -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


@pytest.mark.asyncio
async def test_tiingo_fetch_one_parses_adjusted_ohlcv() -> None:
    inst = _make_inst()
    payload = [
        {
            "date": "2026-01-02",
            "adjOpen": 10.0,
            "adjHigh": 10.5,
            "adjLow": 9.5,
            "adjClose": 10.1,
            "volume": 1000,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "AAPL" in str(request.url)
        assert "Token testtok" in request.headers.get("Authorization", "")
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = TiingoBarFetcher("testtok", max_concurrent=1)
        out = await fetcher._fetch_one(  # noqa: SLF001
            client,
            inst,
            date(2026, 1, 1),
            date(2026, 1, 3),
        )
    assert len(out) == 1
    assert out[0].close == Decimal("10.1")
    assert out[0].bar_seconds == 86400
    assert out[0].instrument_id == inst.instrument_id


@pytest.mark.asyncio
async def test_tiingo_call_empty_instruments() -> None:
    fetcher = TiingoBarFetcher("tok")
    out = await fetcher([], date(2026, 1, 1), date(2026, 1, 2))
    assert out == []


def test_tiingo_requires_token() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TiingoBarFetcher("  ")


# ---------------------------------------------------------------------------
# DataCircuitBreaker unit tests
# ---------------------------------------------------------------------------


def test_data_circuit_breaker_opens_after_threshold() -> None:
    cb = DataCircuitBreaker("test", failure_threshold=3, open_seconds=60.0)
    assert not cb.is_open()
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open()  # below threshold
    cb.record_failure()
    assert cb.is_open()  # threshold reached


def test_data_circuit_breaker_resets_on_success() -> None:
    cb = DataCircuitBreaker("test", failure_threshold=2, open_seconds=60.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    # Simulate expiry by backdating the open_until timestamp.
    cb._open_until = time.monotonic() - 1  # noqa: SLF001
    assert not cb.is_open()  # window expired — circuit re-closes on check
    cb.record_success()
    assert not cb.is_open()
    assert cb._failures == 0  # noqa: SLF001


def test_data_circuit_breaker_auto_closes_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = DataCircuitBreaker("test", failure_threshold=1, open_seconds=30.0)
    cb.record_failure()
    assert cb.is_open()
    # Advance virtual time past the open window.
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 31.0)
    assert not cb.is_open()


# ---------------------------------------------------------------------------
# Circuit breaker integration tests for TiingoBarFetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tiingo_circuit_breaker_skips_when_open() -> None:
    inst = _make_inst()
    fetcher = TiingoBarFetcher("tok")

    # Trip the circuit breaker directly without making HTTP calls.
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
    assert call_count == 0  # _fetch_one must not be called when circuit is open


@pytest.mark.asyncio
async def test_tiingo_circuit_breaker_records_failure_on_server_error() -> None:
    inst = _make_inst()
    fetcher = TiingoBarFetcher("tok")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    import httpx as _httpx

    async with _httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert out == []
    assert fetcher._cb._failures > 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_tiingo_circuit_breaker_records_success_on_valid_bars() -> None:
    inst = _make_inst()
    fetcher = TiingoBarFetcher("tok")
    # Prime with a failure so we can verify reset.
    fetcher._cb.record_failure()  # noqa: SLF001

    payload = [
        {
            "date": "2026-01-02",
            "adjOpen": 10.0,
            "adjHigh": 10.5,
            "adjLow": 9.5,
            "adjClose": 10.1,
            "volume": 1000,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await fetcher._fetch_one(client, inst, date(2026, 1, 1), date(2026, 1, 5))  # noqa: SLF001

    assert len(out) == 1
    assert fetcher._cb._failures == 0  # noqa: SLF001
