"""Polygon historical 1-minute aggregate adapter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from quant_platform.core.domain.market_data import MarketBar, VendorBarBatch
from quant_platform.services.data_service.intraday import (
    INTRADAY_BAR_SECONDS,
    canonical_intraday_bar_id,
)

PolygonQueryValue = str | int | float | bool | None
PolygonQueryParams = Mapping[str, PolygonQueryValue]

if TYPE_CHECKING:
    import uuid

log = structlog.get_logger(__name__)


class PolygonHistoricalBarVendorAdapter:
    """HistoricalBarVendorAdapter for Polygon stock minute aggregates."""

    def __init__(
        self,
        *,
        api_key: str,
        symbol_by_instrument_id: Mapping[uuid.UUID, str],
        base_url: str = "https://api.polygon.io",
        timeout_seconds: float = 30.0,
        max_concurrent: int = 4,
        client: httpx.AsyncClient | None = None,
        retry_sleep_seconds: float = 0.25,
        max_retries: int = 2,
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("PolygonHistoricalBarVendorAdapter requires polygon_api_key")
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must be >= 0")
        self._api_key = api_key.strip()
        self._symbols = {
            iid: symbol.strip().upper() for iid, symbol in symbol_by_instrument_id.items()
        }
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._sem = asyncio.Semaphore(max_concurrent)
        self._client = client
        self._retry_sleep_seconds = retry_sleep_seconds
        self._max_retries = max_retries
        self._min_request_interval_seconds = min_request_interval_seconds
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def fetch_bars(
        self,
        instruments: list[uuid.UUID],
        start: datetime,
        end: datetime,
        bar_seconds: int,
        *,
        as_of: datetime,
    ) -> VendorBarBatch:
        """Fetch adjusted 1-minute bars from Polygon and canonicalize them."""
        if bar_seconds != INTRADAY_BAR_SECONDS:
            raise ValueError("Polygon intraday adapter supports only 1-minute bars")
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        fetched_at = _ensure_utc(as_of)
        missing_symbols = [str(iid) for iid in instruments if iid not in self._symbols]
        bars: list[MarketBar] = []
        request_ids: list[str] = []
        source_uris: list[str] = []

        async def _run(client: httpx.AsyncClient) -> None:
            tasks = [
                self._fetch_instrument(client, iid, start_utc, end_utc)
                for iid in instruments
                if iid in self._symbols
            ]
            for result_bars, result_request_ids, result_source in await asyncio.gather(*tasks):
                bars.extend(result_bars)
                request_ids.extend(result_request_ids)
                source_uris.append(result_source)

        if self._client is not None:
            await _run(self._client)
        else:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                await _run(client)

        canonical = tuple(sorted(bars, key=lambda bar: (str(bar.instrument_id), bar.timestamp)))
        coverage = _coverage(canonical)
        coverage.update(
            {
                "vendor": "polygon",
                "requested_instruments": len(instruments),
                "missing_symbol_instruments": missing_symbols,
                "request_ids": sorted(set(request_ids)),
                "source_uri_count": len(source_uris),
            }
        )
        return VendorBarBatch(
            vendor="polygon",
            source_uri=";".join(sorted(set(source_uris))) or "polygon://stocks/minute-aggs",
            fetched_at=fetched_at,
            bar_seconds=bar_seconds,
            bars=canonical,
            coverage=coverage,
        )

    async def _fetch_instrument(
        self,
        client: httpx.AsyncClient,
        instrument_id: uuid.UUID,
        start: datetime,
        end: datetime,
    ) -> tuple[list[MarketBar], list[str], str]:
        symbol = self._symbols[instrument_id]
        url = (
            f"{self._base_url}/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start.date().isoformat()}/{end.date().isoformat()}"
        )
        source_uri = (
            f"polygon://stocks/aggs/{symbol}/1/minute?"
            f"from={start.isoformat()}&to={end.isoformat()}&adjusted=true&sort=asc&limit=50000"
        )
        params: dict[str, PolygonQueryValue] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        bars: list[MarketBar] = []
        request_ids: list[str] = []
        next_url: str | None = url
        next_params: PolygonQueryParams | None = params
        while next_url:
            body = await self._request_json(client, next_url, next_params)
            request_id = body.get("request_id")
            if request_id:
                request_ids.append(str(request_id))
            for row in body.get("results") or []:
                bar = _parse_polygon_row(instrument_id, row)
                if bar is not None and start <= bar.timestamp <= end:
                    bars.append(bar)
            raw_next = body.get("next_url")
            next_url = str(raw_next) if raw_next else None
            next_params = None
        return bars, request_ids, source_uri

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: PolygonQueryParams | None,
    ) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            await self._pace_request()
            async with self._sem:
                response = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Polygon response must be a JSON object")
                return payload
            if attempt >= self._max_retries:
                response.raise_for_status()
            await asyncio.sleep(self._retry_delay(response, attempt))
        raise RuntimeError("unreachable Polygon retry loop")

    async def _pace_request(self) -> None:
        if self._min_request_interval_seconds <= 0:
            return
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            delay = self._min_request_interval_seconds - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_at = time.monotonic()

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        raw_retry_after = response.headers.get("retry-after")
        try:
            retry_after = float(raw_retry_after) if raw_retry_after is not None else 0.0
        except ValueError:
            retry_after = 0.0
        return max(
            retry_after,
            self._min_request_interval_seconds,
            self._retry_sleep_seconds * (attempt + 1),
        )


def _parse_polygon_row(instrument_id: uuid.UUID, row: object) -> MarketBar | None:
    if not isinstance(row, Mapping):
        return None
    try:
        timestamp = datetime.fromtimestamp(int(row["t"]) / 1000, tz=UTC)
        open_ = Decimal(str(row["o"]))
        high = Decimal(str(row["h"]))
        low = Decimal(str(row["l"]))
        close = Decimal(str(row["c"]))
        volume = int(row.get("v", 0))
        raw_vwap = row.get("vw")
        vwap = None if raw_vwap is None else Decimal(str(raw_vwap))
        return MarketBar(
            bar_id=canonical_intraday_bar_id(instrument_id, timestamp, INTRADAY_BAR_SECONDS),
            instrument_id=instrument_id,
            timestamp=timestamp,
            bar_seconds=INTRADAY_BAR_SECONDS,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            is_complete=True,
        )
    except (KeyError, TypeError, ValueError):
        log.debug("polygon_intraday.row_skip", instrument_id=str(instrument_id), row=row)
        return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coverage(bars: tuple[MarketBar, ...]) -> dict[str, object]:
    rows_by_instrument: dict[str, int] = {}
    for bar in bars:
        key = str(bar.instrument_id)
        rows_by_instrument[key] = rows_by_instrument.get(key, 0) + 1
    return {
        "row_count": len(bars),
        "instrument_count": len(rows_by_instrument),
        "rows_by_instrument": rows_by_instrument,
        "start_at": min((bar.timestamp.isoformat() for bar in bars), default=None),
        "end_at": max((bar.timestamp.isoformat() for bar in bars), default=None),
    }
