"""News callbacks for the IB wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    from _thread import LockType

    RawHistoricalNewsHeadline = tuple[str, str, str, str]
    RawNewsArticle = tuple[int, str]


class IBNewsCallbackMixin:
    """Historical news and article-body callbacks used by ``_IBWrapper``."""

    _historical_news: dict[int, list[RawHistoricalNewsHeadline]]
    _historical_news_futures: dict[int, asyncio.Future[list[RawHistoricalNewsHeadline]]]
    _lifecycle_lock: LockType
    _news_article_futures: dict[int, asyncio.Future[RawNewsArticle]]

    def _resolve(self, future: asyncio.Future[Any], value: object) -> None:
        raise NotImplementedError

    def historicalNews(
        self,
        requestId: int,
        time: str,
        providerCode: str,
        articleId: str,
        headline: str,
    ) -> None:
        with self._lifecycle_lock:
            entries = self._historical_news.setdefault(requestId, [])
            entries.append((str(time), str(providerCode), str(articleId), str(headline)))

    def historicalNewsEnd(self, requestId: int, hasMore: bool) -> None:
        del hasMore
        with self._lifecycle_lock:
            entries = self._historical_news.pop(requestId, [])
            future = self._historical_news_futures.pop(requestId, None)
        if future is not None:
            self._resolve(future, entries)

    def newsArticle(self, requestId: int, articleType: int, articleText: str) -> None:
        with self._lifecycle_lock:
            future = self._news_article_futures.pop(requestId, None)
        if future is not None:
            self._resolve(future, (int(articleType), str(articleText)))


__all__ = ["IBNewsCallbackMixin"]
