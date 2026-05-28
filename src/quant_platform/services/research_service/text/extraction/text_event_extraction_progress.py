"""Heartbeat task that writes periodic extraction progress JSON.

The text-event extraction loop spawns ``heartbeat_loop`` as an asyncio task so a
long-running backfill can be monitored without tailing structured logs. The
operator reads the file via ``scripts/extract_status.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


async def heartbeat_loop(
    status_file: Path,
    state: dict[str, int | float | str],
    started_monotonic: float,
    stop: asyncio.Event,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    """Write progress every ``interval`` seconds until ``stop`` is set."""
    while not stop.is_set():
        write_status(status_file, state, started_monotonic, terminal=False)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            continue


def write_status(
    status_file: Path | None,
    state: dict[str, int | float | str],
    started_monotonic: float,
    *,
    terminal: bool,
) -> None:
    """Render and atomically write a single progress snapshot."""
    if status_file is None:
        return
    elapsed = max(1e-6, time.monotonic() - started_monotonic)
    extracted = int(state.get("extracted", 0))
    total = int(state.get("total_events", 0))
    rate_per_minute = extracted / elapsed * 60.0 if elapsed > 0 else 0.0
    remaining = max(0, total - extracted - int(state.get("skipped_duplicates", 0)))
    eta_seconds = remaining / (rate_per_minute / 60.0) if rate_per_minute > 0 else None
    payload: dict[str, object] = {
        "started_at": state.get("started_at"),
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_events": total,
        "extracted": extracted,
        "skipped_macro": int(state.get("skipped_macro", 0)),
        "skipped_missing": int(state.get("skipped_missing", 0)),
        "skipped_document_role": int(state.get("skipped_document_role", 0)),
        "skipped_duplicates": int(state.get("skipped_duplicates", 0)),
        "failed": int(state.get("failed", 0)),
        "in_flight": int(state.get("in_flight", 0)),
        "rate_per_minute": round(rate_per_minute, 3),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "terminal": terminal,
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(status_file)


__all__ = ["DEFAULT_HEARTBEAT_INTERVAL_SECONDS", "heartbeat_loop", "write_status"]
