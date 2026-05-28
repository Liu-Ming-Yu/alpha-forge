"""Polygon.io end-of-day prices as a :class:`BarFetcher` for daily ingest.

Used as a tertiary data source (after IB and Tiingo) to satisfy the
three-vendor dataset quorum requirement (R-DAT-04).  Requires a Polygon
API key (``QP__DATA_INGEST__POLYGON_API_KEY``).

Uses the Polygon ``/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}``
endpoint which returns adjusted OHLCV data.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import structlog

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.support.circuit_breaker import DataCircuitBreaker

if TYPE_CHECKING:
    from quant_platform.core.domain.instruments import Instrument

log = structlog.get_logger(__name__)

_POLYGON_BAR_NAMESPACE = uuid.UUID("018f1a2b-3c4d-7e5f-9a0b-1c2d3e4f5a6b")

_CB_FAILURE_THRESHOLD = 5
_CB_OPEN_SECONDS = 120.0


def _bar_id_v5(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    bar_seconds: int,
) -> uuid.UUID:
    key = f"{instrument_id!s}|{timestamp.isoformat()}|{bar_seconds}"
    return uuid.uuid5(_POLYGON_BAR_NAMESPACE, key)


class PolygonDailyBarFetcher:
    """Fetch daily EOD prices from the Polygon REST API (adjusted OHLCV).

    Implements the same callable signature as ``TiingoBarFetcher`` so it
    can be used as a drop-in secondary in ``FailoverBarFetcher``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        bar_seconds: int = 86400,
        base_url: str = "https://api.polygon.io",
        max_concurrent: int = 4,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("PolygonDailyBarFetcher requires a non-empty API key")
        self._api_key = api_key.strip()
        self._bar_seconds = bar_seconds
        self._base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._timeout_seconds = timeout_seconds
        self._cb = DataCircuitBreaker(
            name="polygon",
            failure_threshold=_CB_FAILURE_THRESHOLD,
            open_seconds=_CB_OPEN_SECONDS,
        )

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        inst: Instrument,
        start: date,
        end: date,
    ) -> list[MarketBar]:
        ticker = inst.symbol.replace(".", "-").upper()
        url = (
            f"{self._base_url}/v2/aggs/ticker/{ticker}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}"
        )
        params: dict[str, str] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self._api_key,
        }

        _retry_delays = (1.0, 2.0)
        r: httpx.Response | None = None
        for attempt, delay in enumerate([0.0, *_retry_delays]):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with self._sem:
                    r = await client.get(url, params=params)
            except OSError as exc:
                log.warning(
                    "polygon_daily_bar_fetcher.request_error",
                    symbol=inst.symbol,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < len(_retry_delays):
                    continue
                self._cb.record_failure()
                return []
            if r.status_code not in (429, 503):
                break
            log.warning(
                "polygon_daily_bar_fetcher.retryable_error",
                symbol=inst.symbol,
                attempt=attempt,
                status=r.status_code,
            )

        if r is None:
            self._cb.record_failure()
            return []
        if r.status_code == 404:
            log.warning("polygon_daily_bar_fetcher.ticker_not_found", symbol=inst.symbol)
            return []
        if r.status_code != 200:
            log.warning(
                "polygon_daily_bar_fetcher.http_error",
                symbol=inst.symbol,
                status=r.status_code,
            )
            self._cb.record_failure()
            return []

        try:
            body = r.json()
        except Exception as exc:
            log.warning(
                "polygon_daily_bar_fetcher.bad_json",
                symbol=inst.symbol,
                error=str(exc),
            )
            return []

        results = body.get("results") if isinstance(body, dict) else None
        if not results or not isinstance(results, list):
            return []

        out: list[MarketBar] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            # Polygon returns epoch-milliseconds in the 't' field.
            t_ms = row.get("t")
            if t_ms is None:
                continue
            try:
                ts = datetime.fromtimestamp(int(t_ms) / 1000, tz=UTC)
            except (TypeError, ValueError, OSError):
                continue
            day = ts.date()
            if not (start <= day <= end):
                continue
            try:
                o = Decimal(str(row["o"]))
                h = Decimal(str(row["h"]))
                lo = Decimal(str(row["l"]))
                c = Decimal(str(row["c"]))
                vol = int(row.get("v", 0) or 0)
            except (KeyError, TypeError, ValueError) as exc:
                log.debug(
                    "polygon_daily_bar_fetcher.row_skip",
                    symbol=inst.symbol,
                    day=str(day),
                    error=str(exc),
                )
                continue
            bar = MarketBar(
                bar_id=_bar_id_v5(inst.instrument_id, ts, self._bar_seconds),
                instrument_id=inst.instrument_id,
                timestamp=ts,
                bar_seconds=self._bar_seconds,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=vol,
                is_complete=True,
            )
            out.append(bar)
        if out:
            self._cb.record_success()
        return out

    async def __call__(
        self,
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        if not instruments or end < start:
            return []
        if self._cb.is_open():
            log.warning(
                "polygon_daily_bar_fetcher.circuit_open_skip",
                instruments=len(instruments),
            )
            return []
        all_bars: list[MarketBar] = []
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            for inst in instruments:
                bars = await self._fetch_one(client, inst, start, end)
                all_bars.extend(bars)
        log.info(
            "polygon_daily_bar_fetcher.complete",
            instruments=len(instruments),
            bars=len(all_bars),
        )
        return all_bars
