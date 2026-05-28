"""Pure helpers shared by the text-event extraction loop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.services.research_service.text.extraction.text_event_extraction import (
        TextEventExtractionTarget,
    )


def events_for_targets(
    events: list[TextEvent],
    source_targets: Sequence[TextEventExtractionTarget] | None,
) -> tuple[list[TextEvent], list[TextEventExtractionTarget]]:
    """Restrict events to those listed by a governed source manifest."""
    if source_targets is None:
        return events, []
    by_id = {event.event_id: event for event in events}
    scoped: list[TextEvent] = []
    missing: list[TextEventExtractionTarget] = []
    seen: set[uuid.UUID] = set()
    for target in source_targets:
        if target.event_id in seen:
            continue
        seen.add(target.event_id)
        event = by_id.get(target.event_id)
        if event is None:
            missing.append(target)
        else:
            scoped.append(event)
    return scoped, missing


def target_for_event(
    source_targets: Sequence[TextEventExtractionTarget] | None,
    event_id: uuid.UUID,
) -> TextEventExtractionTarget | None:
    if source_targets is None:
        return None
    for target in source_targets:
        if target.event_id == event_id:
            return target
    return None


def read_event_content(artifact_uri: str) -> str:
    path = Path(artifact_uri)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def matches_document_role(event: object, document_role: str) -> bool:
    if document_role == "all":
        return True
    metadata = getattr(event, "metadata", {}) or {}
    raw = str(metadata.get("is_primary_document", "")).strip().lower()
    is_primary = raw in {"1", "true", "yes"}
    if document_role == "primary":
        return is_primary
    return raw in {"0", "false", "no"}


def failure_detail(
    *,
    event: TextEvent | None,
    target: TextEventExtractionTarget | None,
    reason: str,
    error_class: str,
) -> dict[str, object]:
    """Build a normalised JSON-safe failure record."""
    metadata: dict[str, Any] = dict(getattr(event, "metadata", {}) or {})
    return {
        "event_id": str(target.event_id if target is not None else getattr(event, "event_id", "")),
        "error_class": error_class,
        "reason": reason,
        "symbol": _coalesce(target.symbol if target else "", metadata.get("symbol", "")),
        "instrument_id": _coalesce(
            str(target.instrument_id) if target and target.instrument_id else "",
            str(getattr(event, "instrument_id", "") or ""),
        ),
        "occurred_at": _coalesce(
            target.occurred_at.isoformat() if target and target.occurred_at else "",
            getattr(getattr(event, "occurred_at", None), "isoformat", lambda: "")(),
        ),
        "source_uri": _coalesce(
            target.source_uri if target else "",
            getattr(event, "source_uri", ""),
        ),
        "artifact_uri": _coalesce(
            target.artifact_uri if target else "",
            getattr(event, "artifact_uri", ""),
        ),
        "accession_number": _coalesce(
            target.accession_number if target else "",
            metadata.get("accession_number", ""),
        ),
        "form_type": _coalesce(target.form_type if target else "", metadata.get("form_type", "")),
        "document_type": _coalesce(
            target.document_type if target else "",
            metadata.get("document_type", ""),
        ),
        "document_name": _coalesce(
            target.document_name if target else "",
            metadata.get("document_name", ""),
        ),
        "document_description": _coalesce(
            target.document_description if target else "",
            metadata.get("document_description", ""),
        ),
        "content_hash": _coalesce(
            target.content_hash if target else "",
            metadata.get("content_hash", ""),
        ),
    }


def _coalesce(primary: object, fallback: object) -> str:
    primary_text = str(primary or "")
    return primary_text if primary_text else str(fallback or "")


__all__ = [
    "events_for_targets",
    "failure_detail",
    "matches_document_role",
    "read_event_content",
    "target_for_event",
]
