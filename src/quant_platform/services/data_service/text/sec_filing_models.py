"""SEC filing ingestion data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping


@dataclass(frozen=True)
class SECFilingRecord:
    """One SEC filing payload supplied by an upstream downloader."""

    accession_number: str
    cik: str
    form_type: str
    filed_at: datetime
    source_uri: str
    text_content: str
    instrument_id: uuid.UUID | None = None
    document_type: str = ""
    document_name: str = ""
    document_description: str = ""
    is_primary_document: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SECFilingDownloadSummary:
    """Summary of a SEC filing download pass."""

    requested_symbols: tuple[str, ...]
    forms: tuple[str, ...]
    start: datetime
    end: datetime
    records_downloaded: int
    filings_scanned: int = 0
    primary_documents_downloaded: int = 0
    exhibits_downloaded: int = 0
    document_type_counts: Mapping[str, int] = field(default_factory=dict)
    failed_accessions: tuple[str, ...] = ()
    missing_cik_symbols: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "requested_symbols": list(self.requested_symbols),
            "forms": list(self.forms),
            "start": self.start.astimezone(UTC).isoformat(),
            "end": self.end.astimezone(UTC).isoformat(),
            "records_downloaded": self.records_downloaded,
            "filings_scanned": self.filings_scanned,
            "primary_documents_downloaded": self.primary_documents_downloaded,
            "exhibits_downloaded": self.exhibits_downloaded,
            "document_type_counts": dict(sorted(self.document_type_counts.items())),
            "failed_accessions": list(self.failed_accessions),
            "missing_cik_symbols": list(self.missing_cik_symbols),
        }


@dataclass(frozen=True)
class SECFilingDocument:
    """One document listed under an SEC accession."""

    document_name: str
    document_type: str
    description: str
    source_uri: str
    is_primary_document: bool = False


__all__ = ["SECFilingDocument", "SECFilingDownloadSummary", "SECFilingRecord"]
