"""Shared parsing helpers for CLI and bootstrap entrypoints."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def load_instrument_contracts(path: str | Path) -> dict[uuid.UUID, dict[str, object]]:
    """Load instrument-contract JSON keyed by instrument UUID."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    contracts: dict[uuid.UUID, dict[str, object]] = {}
    for key, value in payload.items():
        try:
            contracts[uuid.UUID(str(key))] = dict(value)
        except ValueError as exc:
            raise SystemExit(f"contracts-file contains invalid UUID key {key!r}: {exc}") from exc
    return contracts


def instrument_lookup_from_contracts(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[str, uuid.UUID]:
    """Return lookup aliases by UUID string and upper-case symbol."""
    lookup: dict[str, uuid.UUID] = {}
    for instrument_id, spec in contracts.items():
        lookup[str(instrument_id)] = instrument_id
        symbol = spec.get("symbol")
        if symbol:
            lookup[str(symbol).upper()] = instrument_id
    return lookup


def symbol_by_instrument_from_contracts(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[uuid.UUID, str]:
    """Return upper-case symbols keyed by instrument UUID."""
    symbols: dict[uuid.UUID, str] = {}
    for instrument_id, spec in contracts.items():
        symbol = spec.get("symbol")
        if symbol:
            symbols[instrument_id] = str(symbol).upper()
    return symbols


def parse_vendor_file(raw: str, *, option_name: str = "--vendor-file") -> tuple[str, Path]:
    """Parse ``vendor=/path`` option values."""
    if "=" not in raw:
        raise SystemExit(f"{option_name} must use vendor=/path/to/file")
    vendor, path = raw.split("=", 1)
    if not vendor.strip() or not path.strip():
        raise SystemExit(f"{option_name} must use vendor=/path/to/file")
    return vendor.strip(), Path(path)


def parse_intraday_decision_times(
    raw_values: list[str],
    start: datetime,
    end: datetime,
) -> tuple[datetime, ...]:
    """Parse ISO datetimes or HH:MM daily decision times into UTC datetimes."""
    start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
    end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)
    parsed: list[datetime] = []
    for raw in raw_values:
        value = raw.strip()
        if "T" in value:
            ts = datetime.fromisoformat(value)
            parsed.append(ts if ts.tzinfo else ts.replace(tzinfo=UTC))
            continue
        try:
            hh, mm = value.split(":", 1)
            t = datetime_time(hour=int(hh), minute=int(mm), tzinfo=UTC)
        except Exception as exc:
            raise SystemExit("--decision-time must be ISO datetime or HH:MM") from exc
        current = start_utc.date()
        while current <= end_utc.date():
            candidate = datetime.combine(current, t)
            if start_utc <= candidate <= end_utc:
                parsed.append(candidate)
            current = current + timedelta(days=1)
    out = tuple(sorted(set(parsed)))
    if not out:
        raise SystemExit("decision times are empty after applying start/end")
    return out
