"""Integration tests for DistributedLock against a real Redis instance."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress

import pytest

from quant_platform.infrastructure.support.distributed_lock import DistributedLock

pytestmark = pytest.mark.integration_durable


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for durable Redis tests")
    return url


def _lock(
    name: str | None = None, *, ttl: int = 10, acquire_timeout: float = 2.0
) -> DistributedLock:
    return DistributedLock(
        redis_url=_redis_url(),
        name=name or f"test-lock-{uuid.uuid4()}",
        ttl_seconds=ttl,
        acquire_timeout_seconds=acquire_timeout,
    )


@pytest.mark.asyncio
async def test_acquire_and_release_round_trip() -> None:
    lock = _lock()
    acquired = await lock.acquire()
    assert acquired is True
    assert lock.lease_lost is False

    import redis.asyncio as aioredis

    client = aioredis.from_url(_redis_url(), decode_responses=True)
    key_exists_before = await client.exists(lock._name)
    assert key_exists_before == 1

    await lock.release()
    key_exists_after = await client.exists(lock._name)
    assert key_exists_after == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_acquire_blocks_second_holder() -> None:
    name = f"test-lock-{uuid.uuid4()}"
    lock1 = _lock(name)
    lock2 = _lock(name, acquire_timeout=0.6)

    acquired1 = await lock1.acquire()
    assert acquired1 is True
    try:
        acquired2 = await lock2.acquire()
        assert acquired2 is False
    finally:
        await lock1.release()


@pytest.mark.asyncio
async def test_context_manager_releases_on_exit() -> None:
    name = f"test-lock-{uuid.uuid4()}"
    lock1 = _lock(name)

    async with lock1:
        pass

    lock2 = _lock(name, acquire_timeout=1.0)
    acquired = await lock2.acquire()
    assert acquired is True
    await lock2.release()


@pytest.mark.asyncio
async def test_lease_renewal_maintains_lock_beyond_ttl() -> None:
    lock = DistributedLock(
        redis_url=_redis_url(),
        name=f"test-renew-{uuid.uuid4()}",
        ttl_seconds=2,
        acquire_timeout_seconds=2.0,
        renew_interval_seconds=0.5,
    )
    acquired = await lock.acquire()
    assert acquired is True
    try:
        await asyncio.sleep(3)

        import redis.asyncio as aioredis

        client = aioredis.from_url(_redis_url(), decode_responses=True)
        still_exists = await client.exists(lock._name)
        await client.aclose()
        assert still_exists == 1
        assert lock.lease_lost is False
    finally:
        await lock.release()


@pytest.mark.asyncio
async def test_lease_loss_detected_when_key_deleted_externally() -> None:
    lock = DistributedLock(
        redis_url=_redis_url(),
        name=f"test-lostlease-{uuid.uuid4()}",
        ttl_seconds=30,
        acquire_timeout_seconds=2.0,
        renew_interval_seconds=0.5,
    )
    acquired = await lock.acquire()
    assert acquired is True
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(_redis_url(), decode_responses=True)
        await client.delete(lock._name)
        await client.aclose()

        await asyncio.sleep(1.0)
        assert lock.lease_lost is True
    finally:
        with suppress(Exception):
            await lock.release()


@pytest.mark.asyncio
async def test_release_when_not_owner_is_safe() -> None:
    name = f"test-notowner-{uuid.uuid4()}"
    lock = _lock(name)
    acquired = await lock.acquire()
    assert acquired is True

    import redis.asyncio as aioredis

    client = aioredis.from_url(_redis_url(), decode_responses=True)
    await client.set(lock._name, "different-holder-id", ex=30)

    await lock.release()

    still_there = await client.exists(lock._name)
    assert still_there == 1
    await client.delete(lock._name)
    await client.aclose()
