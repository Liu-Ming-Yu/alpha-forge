"""Tiingo end-of-day prices as a :class:`BarFetcher` for daily ingest.

Used as a secondary data source when IB historical data is unavailable
(R-DAT-04).  Requires a Tiingo API token
(`QP__DATA_INGEST__TIINGO_API_TOKEN`).  The adapter is HTTP-only;
no additional broker install is required.
"""

from __future__ import annotations

import asyncio
import json
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

# Used to generate deterministic bar_id values for Tiingo-sourced rows.
_TIINGO_BAR_NAMESPACE = uuid.UUID("018f0e8a-0c4d-7c3b-8d2a-0f1e2d3c4b5a")

_CB_FAILURE_THRESHOLD = 5
_CB_OPEN_SECONDS = 120.0


def _bar_id_v5(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    bar_seconds: int,
) -> uuid.UUID:
    key = f"{instrument_id!s}|{timestamp.isoformat()}|{bar_seconds}"
    return uuid.uuid5(_TIINGO_BAR_NAMESPACE, key)


def _parse_rows(
    inst: Instrument,
    body: str | bytes,
) -> list[dict[str, object]]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        log.warning("tiingo_bar_fetcher.bad_json", symbol=inst.symbol, error=str(exc))
        return []
    if not isinstance(data, list):
        log.warning("tiingo_bar_fetcher.unexpected_shape", symbol=inst.symbol)
        return []
    return [row for row in data if isinstance(row, dict)]


class TiingoBarFetcher:
    """Fetch daily EOD prices from the Tiingo REST API (adjusted OHLCV)."""

    def __init__(
        self,
        token: str,
        *,
        bar_seconds: int = 86400,
        max_concurrent: int = 8,
    ) -> None:
        if not token.strip():
            raise ValueError("TiingoBarFetcher requires a non-empty API token")
        self._token = token.strip()
        self._bar_seconds = bar_seconds
        self._sem = asyncio.Semaphore(max_concurrent)
        self._cb = DataCircuitBreaker(
            name="tiingo",
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
        ticker = inst.symbol.replace(".", "-")
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        }
        headers = {"Authorization": f"Token {self._token}"}

        _retry_delays = (2.0, 4.0)
        r: httpx.Response | None = None
        for attempt, delay in enumerate([0.0, *_retry_delays]):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with self._sem:
                    r = await client.get(url, params=params, headers=headers)
            except OSError as exc:
                log.warning(
                    "tiingo_bar_fetcher.request_error",
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
                "tiingo_bar_fetcher.retryable_error",
                symbol=inst.symbol,
                attempt=attempt,
                status=r.status_code,
            )

        if r is None:
            self._cb.record_failure()
            return []
        if r.status_code == 404:
            log.warning("tiingo_bar_fetcher.ticker_not_found", symbol=inst.symbol)
            return []
        if r.status_code != 200:
            log.warning(
                "tiingo_bar_fetcher.http_error",
                symbol=inst.symbol,
                status=r.status_code,
            )
            self._cb.record_failure()
            return []

        rows = _parse_rows(inst, r.content)
        out: list[MarketBar] = []
        for row in rows:
            d_raw = row.get("date")
            if not d_raw or not isinstance(d_raw, str):
                continue
            day = date.fromisoformat(d_raw[:10])
            if not (start <= day <= end):
                continue
            # Adjusted series — consistent with Parquet split/dividend-adjusted read path
            a_o = row.get("adjOpen")
            a_h = row.get("adjHigh")
            a_l = row.get("adjLow")
            a_c = row.get("adjClose")
            if any(x is None for x in (a_o, a_h, a_l, a_c)):
                a_o = row.get("open")
                a_h = row.get("high")
                a_l = row.get("low")
                a_c = row.get("close")
            if any(x is None for x in (a_o, a_h, a_l, a_c)):
                continue
            vol_raw = row.get("volume", 0)
            try:
                vol = int(float(str(vol_raw))) if vol_raw is not None else 0
            except (TypeError, ValueError):
                vol = 0
            ts = datetime(day.year, day.month, day.day, tzinfo=UTC)
            try:
                o = Decimal(str(a_o))
                h = Decimal(str(a_h))
                low = Decimal(str(a_l))
                c = Decimal(str(a_c))
                bar = MarketBar(
                    bar_id=_bar_id_v5(inst.instrument_id, ts, self._bar_seconds),
                    instrument_id=inst.instrument_id,
                    timestamp=ts,
                    bar_seconds=self._bar_seconds,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=vol,
                    is_complete=True,
                )
            except (ValueError, TypeError) as exc:
                log.debug(
                    "tiingo_bar_fetcher.row_skip",
                    symbol=inst.symbol,
                    day=str(day),
                    error=str(exc),
                )
                continue
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
        if not instruments:
            return []
        if end < start:
            return []
        if self._cb.is_open():
            log.warning(
                "tiingo_bar_fetcher.circuit_open_skip",
                instruments=len(instruments),
            )
            return []
        all_bars: list[MarketBar] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for inst in instruments:
                bars = await self._fetch_one(client, inst, start, end)
                all_bars.extend(bars)
        log.info(
            "tiingo_bar_fetcher.complete",
            instruments=len(instruments),
            bars=len(all_bars),
        )
        return all_bars
