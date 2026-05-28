"""Manifest helpers for governed text-event evidence runs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from quant_platform.services.research_service.text.extraction.text_event_extraction import (
    TextEventExtractionTarget,
)

if TYPE_CHECKING:
    from pathlib import Path


def event_manifest_records(events: Sequence[object]) -> list[dict[str, object]]:
    records = [_event_manifest_record(event) for event in events]
    return sorted(
        records,
        key=lambda record: (
            str(record.get("symbol", "")),
            str(record.get("occurred_at", "")),
            str(record.get("event_id", "")),
        ),
    )


def load_manifest_extraction_targets(
    path: Path,
    *,
    document_role: str,
) -> tuple[tuple[TextEventExtractionTarget, ...], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (), f"source-data manifest unavailable: {exc}"
    if not isinstance(payload, Mapping):
        return (), "source-data manifest must be a JSON object"
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return (
            (),
            "source-data manifest is missing per-event records; rerun "
            "text-events ingest-sec --include-exhibits",
        )

    targets: list[TextEventExtractionTarget] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            return (), f"source-data manifest event[{index}] must be a JSON object"
        if not _record_matches_document_role(raw_event, document_role):
            continue
        target, error = _target_from_record(raw_event, index=index)
        if error:
            return (), error
        targets.append(target)
    if not targets:
        return (), f"source-data manifest has zero events for document_role={document_role}"
    return tuple(targets), ""


def _event_manifest_record(event: object) -> dict[str, object]:
    metadata = getattr(event, "metadata", {}) or {}
    instrument_id = getattr(event, "instrument_id", None)
    occurred_at = getattr(event, "occurred_at", None)
    occurred_at_text = occurred_at.isoformat() if isinstance(occurred_at, datetime) else ""
    record: dict[str, object] = {
        "event_id": str(getattr(event, "event_id", "")),
        "symbol": str(metadata.get("symbol", "")).upper(),
        "instrument_id": str(instrument_id) if instrument_id is not None else None,
        "occurred_at": occurred_at_text,
        "source_uri": str(getattr(event, "source_uri", "")),
        "artifact_uri": str(getattr(event, "artifact_uri", "")),
        "accession_number": str(metadata.get("accession_number", "")),
        "cik": str(metadata.get("cik", "")),
        "form_type": str(metadata.get("form_type", "")),
        "document_type": str(metadata.get("document_type", "")),
        "document_name": str(metadata.get("document_name", "")),
        "document_description": str(metadata.get("document_description", "")),
        "is_primary_document": _metadata_bool(metadata.get("is_primary_document")),
        "content_hash": str(metadata.get("content_hash", "")),
    }
    if metadata.get("source_kind") == "news":
        record.update(
            {
                "provider": str(metadata.get("provider", "")),
                "provider_code": str(metadata.get("provider_code", "")),
                "article_id": str(metadata.get("article_id", "")),
                "headline": str(metadata.get("headline", "")),
                "ingestion_status": str(metadata.get("ingestion_status", "")),
            }
        )
    return record


def _target_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
) -> tuple[TextEventExtractionTarget, str]:
    try:
        event_id = uuid.UUID(str(record["event_id"]))
    except (KeyError, TypeError, ValueError):
        return _empty_target(), f"source-data manifest event[{index}] has invalid event_id"
    instrument_id = _optional_uuid(record.get("instrument_id"))
    if record.get("instrument_id") and instrument_id is None:
        return _empty_target(), f"source-data manifest event[{index}] has invalid instrument_id"
    occurred_at = _optional_datetime(record.get("occurred_at"))
    if record.get("occurred_at") and occurred_at is None:
        return _empty_target(), f"source-data manifest event[{index}] has invalid occurred_at"
    return (
        TextEventExtractionTarget(
            event_id=event_id,
            symbol=str(record.get("symbol", "")).upper(),
            instrument_id=instrument_id,
            occurred_at=occurred_at,
            source_uri=str(record.get("source_uri", "")),
            artifact_uri=str(record.get("artifact_uri", "")),
            accession_number=str(record.get("accession_number", "")),
            form_type=str(record.get("form_type", "")),
            document_type=str(record.get("document_type", "")),
            document_name=str(record.get("document_name", "")),
            document_description=str(record.get("document_description", "")),
            is_primary_document=_metadata_bool(record.get("is_primary_document")),
            content_hash=str(record.get("content_hash", "")),
        ),
        "",
    )


def _record_matches_document_role(record: Mapping[str, Any], document_role: str) -> bool:
    if document_role == "all":
        return True
    is_primary = _metadata_bool(record.get("is_primary_document"))
    return is_primary if document_role == "primary" else not is_primary


def _metadata_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_uuid(value: object) -> uuid.UUID | None:
    if value in {None, ""}:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


def _optional_datetime(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _empty_target() -> TextEventExtractionTarget:
    return TextEventExtractionTarget(event_id=uuid.UUID(int=0))


__all__ = ["event_manifest_records", "load_manifest_extraction_targets"]
