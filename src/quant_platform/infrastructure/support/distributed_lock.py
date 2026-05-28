"""Distributed lock for multi-worker safety of in-process state.

The CashLedger is not thread-safe; concurrent mutation from multiple workers
would corrupt settled/unsettled/reserved balances.  This module provides a
Redis-backed advisory lock that must be held for the duration of any mutating
operation on shared state (strategy cycle, settlement advancement, etc.).

When no Redis URL is configured, the NullLock fallback is used — it always
succeeds immediately, preserving the single-process default behaviour.

Usage::

    lock = create_distributed_lock(settings.storage.redis_url, "strategy-cycle")
    async with lock:
        await run_strategy_cycle(...)
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Final

import structlog

from quant_platform.infrastructure.metrics import (
    record_lease_loss,
    record_lock_operation,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts.redis import AsyncRedisClient

log = structlog.get_logger(__name__)

_LOCK_DEFAULT_TTL_SECONDS = 120
_LOCK_ACQUIRE_TIMEOUT_SECONDS = 30
_LOCK_RETRY_INTERVAL_SECONDS = 0.5
_LOCK_MIN_RENEW_INTERVAL_SECONDS = 1.0

_RELEASE_SCRIPT: Final[str] = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

_RENEW_SCRIPT: Final[str] = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
else
    return 0
end
"""


class DistributedLock:
    """Redis-backed advisory lock using SET NX with TTL.

    Implements the async context manager protocol so callers can use
    ``async with lock:`` for automatic acquisition and release.

    The lock is identified by a (name, holder_id) pair.  The holder_id is
    a random UUID generated at construction time so that only the holder
    that acquired the lock can release it.

    Args:
        redis_url: Redis connection string (``redis://host:port/db``).
        name: Lock name — use a unique string per critical section.
        ttl_seconds: Maximum hold time before Redis auto-expires the key.
        acquire_timeout_seconds: How long to retry before giving up.
    """

    def __init__(
        self,
        redis_url: str,
        name: str,
        ttl_seconds: int = _LOCK_DEFAULT_TTL_SECONDS,
        acquire_timeout_seconds: float = _LOCK_ACQUIRE_TIMEOUT_SECONDS,
        renew_interval_seconds: float | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._name = f"qp:lock:{name}"
        self._ttl = ttl_seconds
        self._timeout = acquire_timeout_seconds
        if renew_interval_seconds is None:
            renew_interval_seconds = max(
                _LOCK_MIN_RENEW_INTERVAL_SECONDS,
                float(ttl_seconds) / 3.0,
            )
        self._renew_interval = min(
            renew_interval_seconds,
            max(_LOCK_MIN_RENEW_INTERVAL_SECONDS, float(ttl_seconds) - 1.0),
        )
        self._holder_id = str(uuid.uuid4())
        self._client: AsyncRedisClient | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self._lease_lost = False

    async def _get_client(self) -> AsyncRedisClient:
        if self._client is None:
            from quant_platform.infrastructure.support.redis_factory import (
                create_async_redis_client,
            )

            self._client = create_async_redis_client(self._redis_url, decode_responses=True)
        return self._client

    async def acquire(self) -> bool:
        client = await self._get_client()
        elapsed = 0.0
        while elapsed < self._timeout:
            acquired = await client.set(
                self._name,
                self._holder_id,
                nx=True,
                ex=self._ttl,
            )
            if acquired:
                self._lease_lost = False
                self._start_renew_loop()
                record_lock_operation(self._name, "acquire", "ok")
                log.debug(
                    "distributed_lock.acquired",
                    name=self._name,
                    ttl_seconds=self._ttl,
                    renew_interval_seconds=self._renew_interval,
                )
                return True
            await asyncio.sleep(_LOCK_RETRY_INTERVAL_SECONDS)
            elapsed += _LOCK_RETRY_INTERVAL_SECONDS

        record_lock_operation(self._name, "acquire", "timeout")
        log.warning(
            "distributed_lock.acquire_timeout",
            name=self._name,
            timeout=self._timeout,
        )
        return False

    async def _renew_once(self) -> bool:
        client = await self._get_client()
        renewed = await client.eval(
            _RENEW_SCRIPT,
            1,
            self._name,
            self._holder_id,
            self._ttl,
        )
        return bool(renewed)

    async def _renew_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._renew_interval)
                renewed = await self._renew_once()
                if not renewed:
                    self._lease_lost = True
                    record_lease_loss(self._name)
                    log.error(
                        "distributed_lock.lease_lost",
                        name=self._name,
                        holder_id=self._holder_id,
                    )
                    return
                record_lock_operation(self._name, "renew", "ok")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - protective path
            self._lease_lost = True
            record_lease_loss(self._name)
            record_lock_operation(self._name, "renew", "error")
            log.error(
                "distributed_lock.renew_error",
                name=self._name,
                error=str(exc),
            )

    def _start_renew_loop(self) -> None:
        if self._renew_task is not None and not self._renew_task.done():
            return
        self._renew_task = asyncio.create_task(self._renew_loop())

    async def _stop_renew_loop(self) -> None:
        if self._renew_task is None:
            return
        task = self._renew_task
        self._renew_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def release(self) -> None:
        await self._stop_renew_loop()
        client = await self._get_client()
        released = await client.eval(
            _RELEASE_SCRIPT,
            1,
            self._name,
            self._holder_id,
        )
        if not released:
            record_lock_operation(self._name, "release", "not_owner")
            log.warning(
                "distributed_lock.release_skipped_not_owner",
                name=self._name,
                holder_id=self._holder_id,
            )
        else:
            record_lock_operation(self._name, "release", "ok")
        log.debug("distributed_lock.released", name=self._name)

    async def aclose(self) -> None:
        """Close the underlying Redis client.

        Long-running supervisors should call this once during shutdown so the
        connection pool is released deterministically. Subsequent operations
        will lazily reconnect.
        """
        await self._stop_renew_loop()
        client = self._client
        self._client = None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    async def __aenter__(self) -> DistributedLock:
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(
                f"Could not acquire distributed lock '{self._name}' within {self._timeout}s"
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()
        if self._lease_lost:
            from quant_platform.core.exceptions import DistributedLockError

            raise DistributedLockError(
                f"distributed lock '{self._name}' lease was lost during the critical section; "
                "local state may be inconsistent with other workers"
            )

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost


class NullLock:
    """No-op lock for single-process deployments without Redis."""

    async def acquire(self) -> bool:
        return True

    async def release(self) -> None:
        pass

    async def aclose(self) -> None:
        return

    async def __aenter__(self) -> NullLock:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    @property
    def lease_lost(self) -> bool:
        return False


def create_distributed_lock(
    redis_url: str,
    name: str,
    ttl_seconds: int = _LOCK_DEFAULT_TTL_SECONDS,
    acquire_timeout_seconds: float = _LOCK_ACQUIRE_TIMEOUT_SECONDS,
    renew_interval_seconds: float | None = None,
) -> DistributedLock | NullLock:
    """Factory that returns a Redis lock if configured, else a NullLock."""
    if redis_url:
        return DistributedLock(
            redis_url=redis_url,
            name=name,
            ttl_seconds=ttl_seconds,
            acquire_timeout_seconds=acquire_timeout_seconds,
            renew_interval_seconds=renew_interval_seconds,
        )
    return NullLock()
