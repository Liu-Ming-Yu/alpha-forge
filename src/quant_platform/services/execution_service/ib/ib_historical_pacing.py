"""Pacing limiter for IB historical market-data requests."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from quant_platform.services.execution_service.stores.pacing_store import HistoricalPacingStore

log = structlog.get_logger(__name__)


class IBHistoricalPacingLimiter:
    """In-memory plus optional durable pacing budget for IB historical data."""

    def __init__(
        self,
        *,
        client_id: int,
        window_seconds: float,
        max_requests: int,
        store: HistoricalPacingStore | None,
    ) -> None:
        self._client_id = client_id
        self._window = window_seconds
        self._max = max_requests
        self._store = store
        self._req_times: list[float] = []
        self._hydrated = False

    async def reserve(self) -> None:
        """Wait until a request slot is available, then reserve it."""
        if self._max <= 0:
            return
        await self.hydrate_if_needed()

        while True:
            now = time.time()
            window_start = now - self._window
            self._req_times = [t for t in self._req_times if t >= window_start]
            if self._store is not None:
                try:
                    await self._store.prune_before(window_start)
                except Exception as exc:  # pragma: no cover - redis hiccup
                    log.warning("ib_pacing.prune_failed", error=str(exc))
            if len(self._req_times) < self._max:
                self._req_times.append(now)
                if self._store is not None:
                    try:
                        await self._store.record(now)
                    except Exception as exc:  # pragma: no cover - redis hiccup
                        log.warning("ib_pacing.record_failed", error=str(exc))
                return
            sleep_for = (self._req_times[0] + self._window) - now
            await asyncio.sleep(max(sleep_for, 0.05))

    async def hydrate_if_needed(self) -> None:
        """Populate in-memory pacing state from the durable store once."""
        if self._hydrated or self._store is None:
            return
        try:
            hydrated = await self._store.hydrate()
        except Exception as exc:  # pragma: no cover - redis hiccup
            log.warning("ib_pacing.hydrate_failed", error=str(exc))
            hydrated = []
        if hydrated:
            self._req_times = sorted(hydrated)
            log.info(
                "ib_pacing.hydrated",
                client_id=self._client_id,
                count=len(hydrated),
            )
        self._hydrated = True
