"""News delegates for the IB gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.services.execution_service.ib.ib_news import IBNewsArticle


class IBNewsPort(Protocol):
    async def fetch_historical_news(
        self,
        *,
        instrument_id: uuid.UUID,
        start: datetime,
        end: datetime,
        provider_codes: tuple[str, ...],
        total_results: int,
        include_article_text: bool,
    ) -> list[IBNewsArticle]: ...


class IBGatewayNewsMixin:
    """Historical-news methods for the IB gateway facade."""

    _news_runtime: object

    async def fetch_historical_news(
        self,
        *,
        instrument_id: uuid.UUID,
        start: datetime,
        end: datetime,
        provider_codes: tuple[str, ...],
        total_results: int = 50,
        include_article_text: bool = True,
    ) -> list[IBNewsArticle]:
        return await self._news.fetch_historical_news(
            instrument_id=instrument_id,
            start=start,
            end=end,
            provider_codes=provider_codes,
            total_results=total_results,
            include_article_text=include_article_text,
        )

    @property
    def _news(self) -> IBNewsPort:
        return cast("IBNewsPort", self._news_runtime)


__all__ = ["IBGatewayNewsMixin"]
