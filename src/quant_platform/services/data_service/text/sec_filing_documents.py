"""SEC accession document discovery and download."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.data_service.text.sec_filing_http import (
    AsyncHTTPClient,
    get_json,
    sec_archive_index_json_url,
    sec_archive_url,
    sec_filing_detail_url,
)
from quant_platform.services.data_service.text.sec_filing_models import (
    SECFilingDocument,
    SECFilingRecord,
)
from quant_platform.services.data_service.text.sec_filing_parsing import (
    clean_sec_document_text,
    infer_document_type_from_name,
    is_preferred_exhibit,
    parse_sec_filing_detail_documents,
)

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import uuid


async def preferred_exhibit_documents(
    *,
    client: AsyncHTTPClient,
    cik: str,
    accession: str,
    primary_document: str,
) -> list[SECFilingDocument]:
    documents = await _filing_detail_documents(
        client=client,
        cik=cik,
        accession=accession,
        primary_document=primary_document,
    )
    if not documents:
        documents = await _filing_index_documents(
            client=client,
            cik=cik,
            accession=accession,
            primary_document=primary_document,
        )
    return [document for document in documents if is_preferred_exhibit(document)]


async def download_document_records(
    *,
    client: AsyncHTTPClient,
    documents: Sequence[SECFilingDocument],
    accession: str,
    cik: str,
    form: str,
    filed_at: datetime,
    instrument_id: uuid.UUID,
    symbol: str,
) -> list[SECFilingRecord]:
    records: list[SECFilingRecord] = []
    for index, document in enumerate(documents):
        try:
            response = await client.get(document.source_uri)
            response.raise_for_status()
        except Exception as exc:
            log.warning(
                "sec_text_provider.document_download_failed",
                accession=accession,
                document=document.document_name,
                error=str(exc),
            )
            continue
        text_content = clean_sec_document_text(response.text)
        if not text_content.strip():
            continue
        records.append(
            SECFilingRecord(
                accession_number=accession,
                cik=cik,
                form_type=form,
                filed_at=filed_at + timedelta(seconds=index),
                source_uri=document.source_uri,
                text_content=text_content,
                instrument_id=instrument_id,
                document_type=document.document_type,
                document_name=document.document_name,
                document_description=document.description,
                is_primary_document=document.is_primary_document,
                metadata={
                    "symbol": symbol,
                    "primary_document": documents[0].document_name if documents else "",
                },
            )
        )
    return records


async def _filing_detail_documents(
    *,
    client: AsyncHTTPClient,
    cik: str,
    accession: str,
    primary_document: str,
) -> list[SECFilingDocument]:
    try:
        response = await client.get(sec_filing_detail_url(cik, accession))
        response.raise_for_status()
    except Exception:
        return []
    return parse_sec_filing_detail_documents(
        response.text,
        cik=cik,
        accession=accession,
        primary_document=primary_document,
    )


async def _filing_index_documents(
    *,
    client: AsyncHTTPClient,
    cik: str,
    accession: str,
    primary_document: str,
) -> list[SECFilingDocument]:
    try:
        payload = await get_json(client, sec_archive_index_json_url(cik, accession))
    except Exception:
        return []
    directory = payload.get("directory")
    items = directory.get("item") if isinstance(directory, Mapping) else None
    if not isinstance(items, list):
        return []
    documents: list[SECFilingDocument] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name == primary_document:
            continue
        documents.append(
            SECFilingDocument(
                document_name=name,
                document_type=infer_document_type_from_name(name),
                description=name,
                source_uri=sec_archive_url(cik, accession, name),
                is_primary_document=False,
            )
        )
    return documents


__all__ = ["download_document_records", "preferred_exhibit_documents"]
