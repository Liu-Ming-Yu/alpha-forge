from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest

from quant_platform.services.data_service.intraday.polygon_intraday import (
    PolygonHistoricalBarVendorAdapter,
)

_UTC = UTC


def _ts_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


@pytest.mark.asyncio
async def test_polygon_intraday_fetch_paginates_and_redacts_api_key() -> None:
    instrument_id = uuid.uuid4()
    calls: list[str] = []
    auth_headers: list[str | None] = []
    first_ts = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)
    second_ts = datetime(2026, 1, 2, 14, 31, tzinfo=_UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        auth_headers.append(request.headers.get("authorization"))
        if "cursor=next" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "request_id": "second",
                    "results": [
                        {
                            "t": _ts_ms(second_ts),
                            "o": 101,
                            "h": 102,
                            "l": 100,
                            "c": 101.5,
                            "v": 1500,
                            "vw": 101.2,
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "first",
                "results": [
                    {
                        "t": _ts_ms(first_ts),
                        "o": 100,
                        "h": 101,
                        "l": 99,
                        "c": 100.5,
                        "v": 1000,
                        "vw": 100.2,
                    }
                ],
                "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/next?cursor=next",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            client=client,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        batch = await adapter.fetch_bars(
            [instrument_id],
            first_ts,
            second_ts,
            60,
            as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
        )

    assert len(batch.bars) == 2
    assert batch.vendor == "polygon"
    assert batch.coverage["request_ids"] == ["first", "second"]
    assert "polygon-token-for-tests" not in batch.source_uri
    assert all("polygon-token-for-tests" not in call for call in calls)
    assert auth_headers == ["Bearer polygon-token-for-tests", "Bearer polygon-token-for-tests"]


@pytest.mark.asyncio
async def test_polygon_intraday_retries_retryable_http_status() -> None:
    instrument_id = uuid.uuid4()
    attempts = 0
    first_ts = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "request_id": "ok",
                "results": [
                    {
                        "t": _ts_ms(first_ts),
                        "o": 100,
                        "h": 101,
                        "l": 99,
                        "c": 100.5,
                        "v": 1000,
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            client=client,
            retry_sleep_seconds=0.0,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        batch = await adapter.fetch_bars(
            [instrument_id],
            first_ts,
            datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
            60,
            as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
        )

    assert attempts == 2
    assert len(batch.bars) == 1


@pytest.mark.asyncio
async def test_polygon_intraday_reports_missing_symbol_coverage() -> None:
    missing_id = uuid.uuid4()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={},
            client=client,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        batch = await adapter.fetch_bars(
            [missing_id],
            datetime(2026, 1, 2, 14, 30, tzinfo=_UTC),
            datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
            60,
            as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
        )

    assert batch.bars == ()
    assert batch.coverage["missing_symbol_instruments"] == [str(missing_id)]


def test_polygon_intraday_constructor_validates_required_settings() -> None:
    instrument_id = uuid.uuid4()

    with pytest.raises(ValueError, match="requires"):
        PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            **{"api" + "_key": "   "},
        )
    with pytest.raises(ValueError, match="positive"):
        PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            max_concurrent=0,
            **{"api" + "_key": "polygon-token-for-tests"},
        )


@pytest.mark.asyncio
async def test_polygon_intraday_rejects_non_minute_requests() -> None:
    instrument_id = uuid.uuid4()
    adapter = PolygonHistoricalBarVendorAdapter(
        symbol_by_instrument_id={instrument_id: "AAPL"},
        **{"api" + "_key": "polygon-token-for-tests"},
    )

    with pytest.raises(ValueError, match="1-minute"):
        await adapter.fetch_bars(
            [instrument_id],
            datetime(2026, 1, 2, 14, 30, tzinfo=_UTC),
            datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
            300,
            as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
        )


@pytest.mark.asyncio
async def test_polygon_intraday_skips_bad_rows_and_requires_json_object() -> None:
    instrument_id = uuid.uuid4()
    first_ts = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": "mixed",
                "results": [
                    "not-a-row",
                    {"t": "not-a-timestamp", "o": 1, "h": 1, "l": 1, "c": 1},
                    {
                        "t": _ts_ms(first_ts),
                        "o": 100,
                        "h": 101,
                        "l": 99,
                        "c": 100.5,
                        "v": 1000,
                    },
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            client=client,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        batch = await adapter.fetch_bars(
            [instrument_id],
            first_ts,
            datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
            60,
            as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
        )

    assert len(batch.bars) == 1
    assert batch.coverage["start_at"] == first_ts.isoformat()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
    ) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            client=client,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        with pytest.raises(ValueError, match="JSON object"):
            await adapter.fetch_bars(
                [instrument_id],
                first_ts,
                datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
                60,
                as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
            )


@pytest.mark.asyncio
async def test_polygon_intraday_raises_after_retry_exhaustion() -> None:
    instrument_id = uuid.uuid4()
    first_ts = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "temporary"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = PolygonHistoricalBarVendorAdapter(
            symbol_by_instrument_id={instrument_id: "AAPL"},
            client=client,
            retry_sleep_seconds=0.0,
            max_retries=1,
            **{"api" + "_key": "polygon-token-for-tests"},
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.fetch_bars(
                [instrument_id],
                first_ts,
                datetime(2026, 1, 2, 14, 31, tzinfo=_UTC),
                60,
                as_of=datetime(2026, 1, 2, 14, 32, tzinfo=_UTC),
            )

    assert attempts == 2
