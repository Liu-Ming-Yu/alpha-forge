"""Reporting helpers for text-event source manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def instrument_coverage(events: Sequence[object]) -> list[str]:
    return sorted(
        {
            str(getattr(event, "metadata", {}).get("symbol", "")).upper()
            for event in events
            if str(getattr(event, "metadata", {}).get("symbol", "")).strip()
        }
    )


def document_type_counts(events: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        metadata = getattr(event, "metadata", {})
        key = str(metadata.get("document_type", metadata.get("form_type", ""))).upper()
        if not key:
            key = "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def events_by_symbol(events: Sequence[object], *, primary: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    expected = "true" if primary else "false"
    for event in events:
        metadata = getattr(event, "metadata", {})
        if str(metadata.get("is_primary_document", "")).lower() != expected:
            continue
        symbol = str(metadata.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["document_type_counts", "events_by_symbol", "instrument_coverage"]
