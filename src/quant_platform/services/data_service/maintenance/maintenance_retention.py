"""Event-bus retention sweep helper for data maintenance supervision."""

from __future__ import annotations

import time
from typing import Protocol

import structlog

log = structlog.get_logger(__name__)


class EventBusRetentionWorker(Protocol):
    async def sweep_once(self) -> dict[str, int]: ...


async def maybe_sweep_event_bus(
    *,
    worker: EventBusRetentionWorker | None,
    interval_seconds: float,
    last_sweep_monotonic: float,
) -> float:
    """Run the retention sweeper when due and return the last-sweep timestamp."""
    if worker is None or interval_seconds <= 0:
        return last_sweep_monotonic

    now = time.monotonic()
    if now - last_sweep_monotonic < interval_seconds:
        return last_sweep_monotonic
    try:
        await worker.sweep_once()
    except Exception as exc:  # pragma: no cover - advisory only
        log.warning("maintenance_supervisor.retention_failed", error=str(exc))
    return now


__all__ = ["maybe_sweep_event_bus"]
