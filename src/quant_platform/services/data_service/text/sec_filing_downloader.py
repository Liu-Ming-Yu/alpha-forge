"""Governed SEC filing downloader."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from quant_platform.services.data_service.text.sec_filing_documents import (
    download_document_records,
    preferred_exhibit_documents,
)
from quant_platform.services.data_service.text.sec_filing_http import (
    AsyncHTTPClient,
    RateLimitedSECClient,
    get_json,
    sec_archive_url,
    sec_submissions_url,
)
from quant_platform.services.data_service.text.sec_filing_models import (
    SECFilingDocument,
    SECFilingDownloadSummary,
    SECFilingRecord,
)

if TYPE_CHECKING:
    import uuid


async def download_sec_filing_records(
    *,
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    cik_by_symbol: Mapping[str, str],
    user_agent: str,
    start: datetime,
    end: datetime,
    forms: Sequence[str] = ("10-K", "10-Q", "8-K"),
    timeout_seconds: float = 30.0,
    client: AsyncHTTPClient | None = None,
    limit_per_symbol: int | None = None,
    include_exhibits: bool = False,
) -> tuple[list[SECFilingRecord], SECFilingDownloadSummary]:
    """Download recent SEC filings for the supplied contracts."""
    if not user_agent.strip():
        raise ValueError("SEC user agent is required")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware")
    if end < start:
        raise ValueError("end must be >= start")

    normalized_forms = tuple(str(form).upper() for form in forms)
    symbols = tuple(_contract_symbol(contract) for contract in contracts.values())
    missing = tuple(symbol for symbol in symbols if symbol and symbol not in cik_by_symbol)
    headers = {"User-Agent": user_agent.strip(), "Accept-Encoding": "gzip, deflate"}
    if client is None:
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=timeout_seconds) as owned_client:
            records = await _download_records_with_client(
                client=RateLimitedSECClient(cast("AsyncHTTPClient", owned_client)),
                contracts=contracts,
                cik_by_symbol=cik_by_symbol,
                start=start,
                end=end,
                forms=normalized_forms,
                limit_per_symbol=limit_per_symbol,
                include_exhibits=include_exhibits,
            )
    else:
        records = await _download_records_with_client(
            client=client,
            contracts=contracts,
            cik_by_symbol=cik_by_symbol,
            start=start,
            end=end,
            forms=normalized_forms,
            limit_per_symbol=limit_per_symbol,
            include_exhibits=include_exhibits,
        )

    document_type_counts: dict[str, int] = {}
    for record in records:
        key = (record.document_type or record.form_type).upper()
        document_type_counts[key] = document_type_counts.get(key, 0) + 1
    summary = SECFilingDownloadSummary(
        requested_symbols=tuple(symbol for symbol in symbols if symbol),
        forms=normalized_forms,
        start=start,
        end=end,
        records_downloaded=len(records),
        filings_scanned=len({record.accession_number for record in records}),
        primary_documents_downloaded=sum(1 for record in records if record.is_primary_document),
        exhibits_downloaded=sum(1 for record in records if not record.is_primary_document),
        document_type_counts=document_type_counts,
        missing_cik_symbols=missing,
    )
    return records, summary


def load_sec_cik_map(path: Path | str) -> dict[str, str]:
    """Load a governed symbol->CIK mapping from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CIK map must be a JSON object")
    mapping: dict[str, str] = {}
    for symbol, cik in payload.items():
        symbol_key = str(symbol).upper().strip()
        cik_text = str(cik).strip().lstrip("0")
        if not symbol_key or not cik_text.isdigit():
            raise ValueError(f"invalid CIK mapping for {symbol!r}")
        mapping[symbol_key] = cik_text
    return mapping


async def _download_records_with_client(
    *,
    client: AsyncHTTPClient,
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    cik_by_symbol: Mapping[str, str],
    start: datetime,
    end: datetime,
    forms: Sequence[str],
    limit_per_symbol: int | None,
    include_exhibits: bool,
) -> list[SECFilingRecord]:
    records: list[SECFilingRecord] = []
    for instrument_id, contract in contracts.items():
        symbol = _contract_symbol(contract)
        if not symbol or symbol not in cik_by_symbol:
            continue
        cik = cik_by_symbol[symbol]
        filing_rows = _recent_filing_rows(await get_json(client, sec_submissions_url(cik)))
        matched = 0
        for row in filing_rows:
            form = str(row.get("form", "")).upper()
            filed_at = _filing_datetime(
                str(row.get("filingDate", "")),
                str(row.get("acceptanceDateTime", "")),
            )
            if form not in forms or filed_at is None or not (start <= filed_at <= end):
                continue
            accession = str(row.get("accessionNumber", "")).strip()
            primary_document = str(row.get("primaryDocument", "")).strip()
            if not accession or not primary_document:
                continue
            documents = [
                SECFilingDocument(
                    document_name=primary_document,
                    document_type=form,
                    description="primary document",
                    source_uri=sec_archive_url(cik, accession, primary_document),
                    is_primary_document=True,
                )
            ]
            if include_exhibits:
                documents.extend(
                    await preferred_exhibit_documents(
                        client=client,
                        cik=cik,
                        accession=accession,
                        primary_document=primary_document,
                    )
                )
            records.extend(
                await download_document_records(
                    client=client,
                    documents=documents,
                    accession=accession,
                    cik=cik,
                    form=form,
                    filed_at=filed_at,
                    instrument_id=instrument_id,
                    symbol=symbol,
                )
            )
            matched += 1
            if limit_per_symbol is not None and matched >= limit_per_symbol:
                break
    return records


def _recent_filing_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    filings = payload.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if not isinstance(recent, Mapping):
        return []
    forms = _sequence(recent.get("form"))
    accessions = _sequence(recent.get("accessionNumber"))
    filing_dates = _sequence(recent.get("filingDate"))
    acceptance_dates = _sequence(recent.get("acceptanceDateTime"))
    primary_documents = _sequence(recent.get("primaryDocument"))
    count = min(len(forms), len(accessions), len(filing_dates), len(primary_documents))
    return [
        {
            "form": forms[index],
            "accessionNumber": accessions[index],
            "filingDate": filing_dates[index],
            "acceptanceDateTime": acceptance_dates[index] if index < len(acceptance_dates) else "",
            "primaryDocument": primary_documents[index],
        }
        for index in range(count)
    ]


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


def _filing_datetime(raw: str, raw_acceptance: str = "") -> datetime | None:
    if raw_acceptance:
        normalized = raw_acceptance.strip().replace("Z", "+00:00")
        try:
            accepted_at = datetime.fromisoformat(normalized)
        except ValueError:
            accepted_at = None
        if accepted_at is not None:
            if accepted_at.tzinfo is None:
                return accepted_at.replace(tzinfo=UTC)
            return accepted_at.astimezone(UTC)
    try:
        date_value = datetime.fromisoformat(raw).date()
    except ValueError:
        return None
    return datetime(date_value.year, date_value.month, date_value.day, tzinfo=UTC)


def _contract_symbol(contract: Mapping[str, object]) -> str:
    return str(contract.get("symbol", "")).upper().strip()


__all__ = ["download_sec_filing_records", "load_sec_cik_map"]
