"""Content-addressed news text-event ingestion."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.services.data_service.text.sec_filing_parsing import clean_sec_document_text
from quant_platform.services.data_service.text.text_provider_metrics import observe_ingestion_lag

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.core.contracts import TextEventProvider


@dataclass(frozen=True)
class NewsArticleRecord:
    """One vendor news article/headline ready to persist as a text event."""

    vendor: str
    provider_code: str
    article_id: str
    headline: str
    published_at: datetime
    source_uri: str
    article_text: str = ""
    article_type: int = 0
    instrument_id: uuid.UUID | None = None
    symbol: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vendor.strip():
            raise ValueError("vendor must not be empty")
        if not self.provider_code.strip():
            raise ValueError("provider_code must not be empty")
        if not self.article_id.strip():
            raise ValueError("article_id must not be empty")
        if not self.headline.strip():
            raise ValueError("headline must not be empty")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")


class NewsTextProvider:
    """Content-addressed provider for vendor news records."""

    provider_name = "news"

    def __init__(
        self,
        *,
        records: list[NewsArticleRecord] | None = None,
        artifact_root: Path | str | None = None,
        provider_name: str = "news",
    ) -> None:
        self._records = list(records or [])
        self._artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.provider_name = provider_name.strip().lower() or "news"

    async def ingest(self, store: TextEventProvider) -> list[TextEvent]:
        if not self._records:
            return []
        if self._artifact_root is None:
            raise ValueError("artifact_root is required when ingesting news records")

        artifact_root = self._artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        events: list[TextEvent] = []
        for record in self._records:
            event = await self._store_record(store, record, artifact_root=artifact_root)
            events.append(event)
        return events

    async def _store_record(
        self,
        store: TextEventProvider,
        record: NewsArticleRecord,
        *,
        artifact_root: Path,
    ) -> TextEvent:
        text_content = _news_artifact_text(record)
        content_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        vendor = record.vendor.strip().lower()
        dedupe_key = (
            f"news:{vendor}:{record.provider_code}:{record.article_id}:"
            f"{record.instrument_id or ''}:{content_hash}"
        )
        artifact_path = artifact_root / vendor / f"{content_hash}.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not artifact_path.exists():
            artifact_path.write_text(text_content, encoding="utf-8")

        occurred_at = record.published_at.astimezone(UTC)
        event = TextEvent(
            event_id=uuid.uuid5(uuid.NAMESPACE_URL, dedupe_key),
            event_type=TextEventType.NEWS_HEADLINE,
            occurred_at=occurred_at,
            source_uri=record.source_uri,
            artifact_uri=str(artifact_path),
            instrument_id=record.instrument_id,
            metadata=_news_event_metadata(
                record=record,
                content_hash=content_hash,
                dedupe_key=dedupe_key,
                occurred_at=occurred_at,
            ),
        )
        await store.store_event(event)
        observe_ingestion_lag(self.provider_name, event.occurred_at)
        return event


class TWSNewsTextProvider(NewsTextProvider):
    """News provider for Interactive Brokers TWS historical news."""

    provider_name = "tws"

    def __init__(
        self,
        *,
        records: list[NewsArticleRecord] | None = None,
        artifact_root: Path | str | None = None,
    ) -> None:
        super().__init__(
            records=records,
            artifact_root=artifact_root,
            provider_name=self.provider_name,
        )


def _news_artifact_text(record: NewsArticleRecord) -> str:
    parts = [f"Headline: {record.headline.strip()}"]
    body = clean_sec_document_text(record.article_text)
    if body:
        parts.extend(["", body])
    return "\n".join(parts).strip() + "\n"


def _news_event_metadata(
    *,
    record: NewsArticleRecord,
    content_hash: str,
    dedupe_key: str,
    occurred_at: datetime,
) -> dict[str, str]:
    article_text_ready = bool(record.article_text.strip())
    return {
        **record.metadata,
        "provider": record.vendor.strip().lower(),
        "vendor": record.vendor.strip().lower(),
        "source_kind": "news",
        "dedupe_key": dedupe_key,
        "content_hash": content_hash,
        "ingestion_status": "ready" if article_text_ready else "headline_only",
        "provider_code": record.provider_code.strip(),
        "article_id": record.article_id.strip(),
        "headline": record.headline.strip(),
        "symbol": record.symbol.strip().upper(),
        "article_type": str(record.article_type),
        "source_published_at": occurred_at.isoformat(),
    }


__all__ = ["NewsArticleRecord", "NewsTextProvider", "TWSNewsTextProvider"]
