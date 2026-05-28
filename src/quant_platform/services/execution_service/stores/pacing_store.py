"""Durable IB historical-data pacing state.

Before Phase 3.3, ``IBGatewayBrokerGateway._hist_req_times`` lived only
in process memory: a restart cleared the sliding window and the
reconnect could burst past the IB HMDS cap (60 req / 600 s) until the
window refilled itself.  This module persists the window to Redis using
a sorted-set keyed by ``client_id`` so the next process (same client
id) hydrates the window on connect and keeps the pacing budget honest
across restarts.

The abstraction is a :class:`HistoricalPacingStore` protocol with two
concrete implementations:

- :class:`InMemoryPacingStore`: process-local behaviour, used in tests and
  when no Redis URL is configured.
- :class:`RedisPacingStore`: Redis sorted-set backed, exact-membership
  durable; uses wall-clock epoch seconds as the score so restarts see a
  consistent absolute timeline.

Partial mitigation for R-EXE-04 (pacing half); vendor diversity remains
tracked as R-DAT-04.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import AsyncRedisClient

log = structlog.get_logger(__name__)


class HistoricalPacingStore(Protocol):
    """Persistent sliding-window of request timestamps (epoch seconds)."""

    async def record(self, now_epoch: float) -> None: ...
    async def prune_before(self, cutoff_epoch: float) -> None: ...
    async def count(self) -> int: ...
    async def earliest(self) -> float | None: ...
    async def hydrate(self) -> list[float]: ...


class InMemoryPacingStore:
    """Process-local store backed by a sorted list of timestamps."""

    def __init__(self) -> None:
        self._times: list[float] = []

    async def record(self, now_epoch: float) -> None:
        self._times.append(now_epoch)

    async def prune_before(self, cutoff_epoch: float) -> None:
        self._times = [t for t in self._times if t >= cutoff_epoch]

    async def count(self) -> int:
        return len(self._times)

    async def earliest(self) -> float | None:
        return self._times[0] if self._times else None

    async def hydrate(self) -> list[float]:
        return list(self._times)


class RedisPacingStore:
    """Sorted-set backed store keyed per IB client id.

    Each stored request is a member ``<epoch>:<uuid4>`` with score equal
    to the epoch seconds so ``ZRANGEBYSCORE`` / ``ZREMRANGEBYSCORE``
    trim the sliding window exactly.  The uuid suffix keeps members
    unique when two requests land in the same millisecond.
    """

    def __init__(self, redis_url: str, client_id: int, window_seconds: float) -> None:
        self._redis_url = redis_url
        self._client_id = client_id
        self._window = window_seconds
        self._key = f"qp:ibpacing:{client_id}"
        self._redis: AsyncRedisClient | None = None

    async def _conn(self) -> AsyncRedisClient:
        if self._redis is None:
            from quant_platform.services.execution_service.support.redis_factory import (
                create_async_redis_client,
            )

            self._redis = create_async_redis_client(self._redis_url, decode_responses=True)
        return self._redis

    async def record(self, now_epoch: float) -> None:
        member = f"{now_epoch:.6f}:{uuid.uuid4().hex}"
        redis = await self._conn()
        await redis.zadd(self._key, {member: now_epoch})
        # Expire the whole key a little past the window so abandoned
        # client ids don't linger forever.
        await redis.expire(self._key, int(self._window * 3))

    async def prune_before(self, cutoff_epoch: float) -> None:
        redis = await self._conn()
        await redis.zremrangebyscore(self._key, "-inf", f"({cutoff_epoch}")

    async def count(self) -> int:
        redis = await self._conn()
        return int(await redis.zcard(self._key))

    async def earliest(self) -> float | None:
        redis = await self._conn()
        rows = await redis.zrange(self._key, 0, 0, withscores=True)
        if not rows:
            return None
        return float(rows[0][1])

    async def hydrate(self) -> list[float]:
        redis = await self._conn()
        # Trim anything already outside the window so the hydrated
        # buffer matches what ``_reserve_pacing_slot`` would compute.
        cutoff = time.time() - self._window
        await redis.zremrangebyscore(self._key, "-inf", f"({cutoff}")
        rows = await redis.zrange(self._key, 0, -1, withscores=True)
        return [float(score) for _member, score in rows]


def build_pacing_store(
    *, redis_url: str | None, client_id: int, window_seconds: float
) -> HistoricalPacingStore:
    """Pick Redis when ``redis_url`` is set, fall back to in-memory."""
    if redis_url:
        log.info(
            "ib_pacing.backend",
            backend="redis",
            client_id=client_id,
            window_seconds=window_seconds,
        )
        return RedisPacingStore(
            redis_url=redis_url,
            client_id=client_id,
            window_seconds=window_seconds,
        )
    log.info("ib_pacing.backend", backend="in_memory", client_id=client_id)
    return InMemoryPacingStore()
