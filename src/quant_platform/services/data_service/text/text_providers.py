"""Text source provider interfaces and SEC filing ingestion."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.services.data_service.text.news_text_provider import (
    NewsArticleRecord,
    NewsTextProvider,
    TWSNewsTextProvider,
)
from quant_platform.services.data_service.text.sec_filing_downloader import (
    download_sec_filing_records,
    load_sec_cik_map,
)
from quant_platform.services.data_service.text.sec_filing_models import (
    SECFilingDownloadSummary,
    SECFilingRecord,
)
from quant_platform.services.data_service.text.sec_filing_parsing import clean_sec_document_text
from quant_platform.services.data_service.text.text_provider_metrics import observe_ingestion_lag

if TYPE_CHECKING:
    from quant_platform.core.contracts import TextEventProvider


class TextSourceProvider(Protocol):
    """Provider contract for filing, transcript, and news ingestion."""

    provider_name: str

    async def ingest(self, store: TextEventProvider) -> list[TextEvent]:
        """Persist provider events and return the events observed this pass."""
        ...


class SECTextFilingProvider:
    """Content-addressed SEC filing ingester."""

    provider_name = "sec"

    def __init__(
        self,
        *,
        records: list[SECFilingRecord],
        artifact_root: Path | str,
    ) -> None:
        self._records = list(records)
        self._artifact_root = Path(artifact_root)

    async def ingest(self, store: TextEventProvider) -> list[TextEvent]:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        events: list[TextEvent] = []
        for record in self._records:
            event = await self._store_record(store, record)
            events.append(event)
        return events

    async def _store_record(
        self,
        store: TextEventProvider,
        record: SECFilingRecord,
    ) -> TextEvent:
        content_hash = hashlib.sha256(record.text_content.encode("utf-8")).hexdigest()
        document_name = record.document_name or record.metadata.get("primary_document", "")
        document_type = record.document_type or record.form_type
        dedupe_key = f"sec:{record.accession_number}:{document_name}:{content_hash}"
        artifact_path = self._artifact_root / "sec" / f"{content_hash}.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not artifact_path.exists():
            artifact_path.write_text(record.text_content, encoding="utf-8")
        event = TextEvent(
            event_id=uuid.uuid5(uuid.NAMESPACE_URL, dedupe_key),
            event_type=TextEventType.SEC_FILING,
            occurred_at=record.filed_at.astimezone(UTC),
            source_uri=record.source_uri,
            artifact_uri=str(artifact_path),
            instrument_id=record.instrument_id,
            metadata=_event_metadata(
                record=record,
                content_hash=content_hash,
                dedupe_key=dedupe_key,
                document_name=document_name,
                document_type=document_type,
            ),
        )
        await store.store_event(event)
        observe_ingestion_lag(self.provider_name, event.occurred_at)
        return event


def _event_metadata(
    *,
    record: SECFilingRecord,
    content_hash: str,
    dedupe_key: str,
    document_name: str,
    document_type: str,
) -> dict[str, str]:
    return {
        **record.metadata,
        "provider": SECTextFilingProvider.provider_name,
        "dedupe_key": dedupe_key,
        "content_hash": content_hash,
        "ingestion_status": "ready",
        "accession_number": record.accession_number,
        "cik": record.cik,
        "form_type": record.form_type,
        "document_type": document_type,
        "document_name": document_name,
        "document_description": record.document_description,
        "is_primary_document": str(record.is_primary_document).lower(),
        "source_published_at": record.filed_at.astimezone(UTC).isoformat(),
    }


class TranscriptTextProvider:
    """Provider hook for future earnings-call transcript adapters."""

    provider_name = "transcript"

    async def ingest(self, store: TextEventProvider) -> list[TextEvent]:
        del store
        return []


__all__ = [
    "NewsArticleRecord",
    "SECFilingDownloadSummary",
    "SECFilingRecord",
    "NewsTextProvider",
    "SECTextFilingProvider",
    "TextSourceProvider",
    "TranscriptTextProvider",
    "TWSNewsTextProvider",
    "clean_sec_document_text",
    "download_sec_filing_records",
    "load_sec_cik_map",
]
