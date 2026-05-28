"""IB historical-news request helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    RawHistoricalNewsHeadline = tuple[str, str, str, str]
    RawNewsArticle = tuple[int, str]


async def fetch_raw_historical_news(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    req_id: int,
    con_id: int,
    provider_codes: str,
    start_date_time: str,
    end_date_time: str,
    total_results: int,
) -> list[RawHistoricalNewsHeadline]:
    """Issue one IB historical-news request and wait for the end marker."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    loop = asyncio.get_running_loop()
    raw_future: asyncio.Future[list[RawHistoricalNewsHeadline]] = loop.create_future()
    with wrapper_any._lifecycle_lock:
        wrapper_any._historical_news_futures[req_id] = raw_future
        wrapper_any._historical_news[req_id] = []

    client_any.reqHistoricalNews(
        req_id,
        con_id,
        provider_codes,
        start_date_time,
        end_date_time,
        total_results,
        [],
    )

    try:
        return await asyncio.wait_for(raw_future, timeout=timeout)
    except TimeoutError:
        with wrapper_any._lifecycle_lock:
            wrapper_any._historical_news_futures.pop(req_id, None)
            wrapper_any._historical_news.pop(req_id, None)
        raise


async def fetch_raw_news_article(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    req_id: int,
    provider_code: str,
    article_id: str,
) -> RawNewsArticle:
    """Issue one IB article-body request and resolve the callback future."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    loop = asyncio.get_running_loop()
    raw_future: asyncio.Future[RawNewsArticle] = loop.create_future()
    with wrapper_any._lifecycle_lock:
        wrapper_any._news_article_futures[req_id] = raw_future

    client_any.reqNewsArticle(req_id, provider_code, article_id, [])

    try:
        return await asyncio.wait_for(raw_future, timeout=timeout)
    except TimeoutError:
        with wrapper_any._lifecycle_lock:
            wrapper_any._news_article_futures.pop(req_id, None)
        raise


__all__ = ["fetch_raw_historical_news", "fetch_raw_news_article"]
