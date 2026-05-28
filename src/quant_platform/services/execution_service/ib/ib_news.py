"""IB historical-news runtime coordination."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.execution_service.ib.ib_news_sync import (
    fetch_raw_historical_news,
    fetch_raw_news_article,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Mapping

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IBNewsArticle:
    """One TWS historical-news headline plus optional article body."""

    instrument_id: uuid.UUID
    con_id: int
    symbol: str
    provider_code: str
    article_id: str
    headline: str
    published_at: datetime
    raw_published_at: str
    article_type: int
    article_text: str
    article_status: str


class IBNewsRuntime:
    """Fetch TWS historical news and article bodies for configured contracts."""

    def __init__(
        self,
        *,
        client: object,
        wrapper: object,
        timeout: float,
        instrument_contracts: Mapping[uuid.UUID, dict[str, object]],
        require_connected: Callable[[], None],
    ) -> None:
        self._client = client
        self._wrapper = wrapper
        self._timeout = timeout
        self._instrument_contracts = instrument_contracts
        self._require_connected = require_connected
        self._next_req_id = 12000
        self._lock = asyncio.Lock()

    async def fetch_historical_news(
        self,
        *,
        instrument_id: uuid.UUID,
        start: datetime,
        end: datetime,
        provider_codes: tuple[str, ...],
        total_results: int,
        include_article_text: bool,
    ) -> list[IBNewsArticle]:
        """Fetch and locally window TWS historical news for one instrument."""
        self._require_connected()
        spec = self._instrument_contracts.get(instrument_id)
        con_id = _contract_con_id(spec)
        if con_id <= 0:
            log.warning("broker_gateway.news.unmapped", instrument_id=str(instrument_id))
            return []

        provider_codes_text = "+".join(code.strip() for code in provider_codes if code.strip())
        if not provider_codes_text:
            raise ValueError("provider_codes must not be empty")
        if not 1 <= total_results <= 300:
            raise ValueError("total_results must be in [1, 300]")

        start_utc = _aware_utc(start)
        end_utc = _aware_utc(end)
        req_id = await self._reserve_req_id()
        # IB's reqHistoricalNews has subtle quirks across TWS versions:
        #
        #   * (start="", end=value)   → returns 0 rows in TWS 10.x
        #     (the original bug here).
        #   * (start=value, end="")   → returns N rows starting from
        #     ``startDateTime`` in ASCENDING order — meaning a 30-day
        #     window asking for total_results=10 yields the 10
        #     OLDEST items at the start boundary, not the most recent.
        #   * (start=value, end=value) → returns rows from inside the
        #     range but the count interacts oddly with totalResults.
        #   * (start="", end="")      → returns the N most-recent
        #     headlines globally. This is what we want; the local
        #     filter below trims to the requested window.
        #
        # The local filter is authoritative; the TWS bounds only
        # bias the result set. Asking for "most recent N" then
        # filtering to the window is the cleanest path.
        rows = await fetch_raw_historical_news(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            req_id=req_id,
            con_id=con_id,
            provider_codes=provider_codes_text,
            start_date_time="",
            end_date_time="",
            total_results=total_results,
        )
        log.debug(
            "broker_gateway.news.tws_returned",
            instrument_id=str(instrument_id),
            con_id=con_id,
            raw_rows=len(rows),
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            total_results=total_results,
        )

        articles: list[IBNewsArticle] = []
        for raw_time, provider_code, article_id, headline in rows:
            published_at = parse_tws_news_timestamp(raw_time)
            if not (start_utc <= published_at < end_utc):
                continue
            article_type, article_text, article_status = await self._fetch_article_body(
                provider_code=provider_code,
                article_id=article_id,
                include_article_text=include_article_text,
            )
            articles.append(
                IBNewsArticle(
                    instrument_id=instrument_id,
                    con_id=con_id,
                    symbol=str(spec.get("symbol", "") if spec else "").upper(),
                    provider_code=provider_code,
                    article_id=article_id,
                    headline=headline,
                    published_at=published_at,
                    raw_published_at=raw_time,
                    article_type=article_type,
                    article_text=article_text,
                    article_status=article_status,
                )
            )
        return articles

    async def _fetch_article_body(
        self,
        *,
        provider_code: str,
        article_id: str,
        include_article_text: bool,
    ) -> tuple[int, str, str]:
        if not include_article_text:
            return 0, "", "headline_only"
        req_id = await self._reserve_req_id()
        try:
            article_type, article_text = await fetch_raw_news_article(
                client=self._client,
                wrapper=self._wrapper,
                timeout=self._timeout,
                req_id=req_id,
                provider_code=provider_code,
                article_id=article_id,
            )
        except Exception as exc:
            log.warning(
                "broker_gateway.news.article_failed",
                provider_code=provider_code,
                article_id=article_id,
                error=str(exc),
            )
            return 0, "", "headline_only"
        return article_type, article_text, "ready" if str(article_text).strip() else "headline_only"

    async def _reserve_req_id(self) -> int:
        async with self._lock:
            req_id = self._next_req_id
            self._next_req_id += 1
        return req_id


def parse_tws_news_timestamp(raw: str) -> datetime:
    """Parse common TWS historical-news timestamp formats as UTC-aware."""
    value = " ".join(str(raw).strip().replace("Z", "+00:00").split())
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    parsed = datetime.fromisoformat(value)
    return _aware_utc(parsed)


def _contract_con_id(spec: dict[str, object] | None) -> int:
    if not spec:
        return 0
    raw = spec.get("con_id")
    return raw if isinstance(raw, int) and raw > 0 else 0


def _aware_utc(value: datetime) -> datetime:
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC)


def _format_tws_datetime(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


__all__ = ["IBNewsArticle", "IBNewsRuntime", "parse_tws_news_timestamp"]
