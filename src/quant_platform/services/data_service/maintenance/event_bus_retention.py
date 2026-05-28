"""Automated retention sweeper for Redis Streams (Phase 4.2 / R-DAT-01).

Calls ``XTRIM <stream> MINID ~ <cutoff>`` periodically so stream memory
does not grow without bound.  In consumer-group mode, publish-time
``MAXLEN`` is disabled and this worker trims only below the oldest pending
entry across all groups so lagged consumers keep their at-least-once window.

Runs inside :class:`DataMaintenanceSupervisor` so its lifecycle is tied
to the existing maintenance loop; wiring it into the main strategy
cycle would couple retention cadence to rebalance cadence, which is
exactly the coupling the rest of the sprint is trying to eliminate.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.core.contracts.redis import AsyncRedisClient

log = structlog.get_logger(__name__)


class EventBusRetentionWorker:
    """Trim Redis Streams by ``MINID`` on a fixed cadence.

    Stream names are expected to be the full keys the event bus uses
    (e.g. ``qp:events:FillReceived``); callers typically enumerate them
    by scanning Redis or by iterating
    :class:`quant_platform.core.events.DomainEvent` subclasses at
    import time.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        stream_keys: Iterable[str],
        retention_ms: int,
    ) -> None:
        if retention_ms < 0:
            raise ValueError("retention_ms must be >= 0")
        self._redis_url = redis_url
        self._streams = [s for s in stream_keys if s]
        self._retention_ms = retention_ms
        self._redis: AsyncRedisClient | None = None

    async def _client(self) -> AsyncRedisClient:
        if self._redis is None:
            from quant_platform.services.data_service.support.redis_factory import (
                create_async_redis_client,
            )

            self._redis = create_async_redis_client(self._redis_url, decode_responses=True)
        return self._redis

    async def sweep_once(self) -> dict[str, int]:
        """Trim each stream to ``now - retention_ms``.

        Returns a ``{stream: trimmed}`` summary where ``trimmed`` is
        the count reported by ``XTRIM``.  ``0`` can mean either the
        stream is empty or it was already within the retention window.
        ``-1`` signals that the trim raised (logged, not re-raised).
        """
        if self._retention_ms == 0 or not self._streams:
            return {}
        client = await self._client()
        cutoff_ms = int(time.time() * 1000) - self._retention_ms
        result: dict[str, int] = {}
        for stream in self._streams:
            try:
                trim_id = await self._safe_trim_id(client, stream, cutoff_ms)
                trimmed = await client.xtrim(stream, minid=trim_id, approximate=True)
                result[stream] = int(trimmed or 0)
            except Exception as exc:  # pragma: no cover - redis hiccup
                log.warning(
                    "event_bus_retention.xtrim_failed",
                    stream=stream,
                    error=str(exc),
                )
                result[stream] = -1
        if any(v > 0 for v in result.values()):
            log.info(
                "event_bus_retention.sweep",
                cutoff_ms=cutoff_ms,
                trimmed=result,
            )
        return result

    async def _safe_trim_id(self, client: AsyncRedisClient, stream: str, cutoff_ms: int) -> str:
        cutoff_id = f"{cutoff_ms}-0"
        oldest_pending = await self._oldest_pending_id(client, stream)
        if oldest_pending is None:
            return cutoff_id
        if _stream_id_lt(oldest_pending, cutoff_id):
            return oldest_pending
        return cutoff_id

    async def _oldest_pending_id(self, client: AsyncRedisClient, stream: str) -> str | None:
        try:
            groups = await client.xinfo_groups(stream)
        except Exception:
            return None
        oldest: str | None = None
        for group in groups or []:
            pending = int(group.get("pending", 0))
            if pending <= 0:
                continue
            group_name = str(group.get("name", ""))
            if not group_name:
                continue
            entries = await client.xpending_range(
                name=stream,
                groupname=group_name,
                min="-",
                max="+",
                count=1,
            )
            if not entries:
                continue
            message_id = str(entries[0].get("message_id"))
            if oldest is None or _stream_id_lt(message_id, oldest):
                oldest = message_id
        return oldest


def _stream_id_lt(left: str, right: str) -> bool:
    left_ms, left_seq = _parse_stream_id(left)
    right_ms, right_seq = _parse_stream_id(right)
    return (left_ms, left_seq) < (right_ms, right_seq)


def _parse_stream_id(value: str) -> tuple[int, int]:
    ms, _, seq = value.partition("-")
    return int(ms), int(seq or "0")
