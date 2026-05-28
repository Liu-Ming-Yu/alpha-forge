"""Integration tests for RedisPacingStore against a real Redis instance."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from quant_platform.services.execution_service.stores.pacing_store import RedisPacingStore

pytestmark = pytest.mark.integration_durable


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for durable Redis tests")
    return url


def _store(window_seconds: float = 600.0) -> RedisPacingStore:
    return RedisPacingStore(
        redis_url=_redis_url(),
        client_id=int(uuid.uuid4()) % 10000 + 10000,
        window_seconds=window_seconds,
    )


@pytest.mark.asyncio
async def test_record_and_count() -> None:
    store = _store()
    now = time.time()
    await store.record(now)
    await store.record(now + 1)
    await store.record(now + 2)
    count = await store.count()
    assert count == 3


@pytest.mark.asyncio
async def test_prune_before() -> None:
    store = _store()
    now = time.time()
    await store.record(now)
    await store.record(now + 10)
    await store.record(now + 20)

    await store.prune_before(now + 15)
    count = await store.count()
    assert count == 1


@pytest.mark.asyncio
async def test_hydrate_returns_within_window() -> None:
    store = _store(window_seconds=30)
    now = time.time()
    await store.record(now - 60)
    await store.record(now - 50)
    await store.record(now)
    await store.record(now + 1)
    await store.record(now + 2)

    hydrated = await store.hydrate()
    for t in hydrated:
        assert t >= now - 30


@pytest.mark.asyncio
async def test_ttl_set_on_key() -> None:
    store = _store(window_seconds=60)
    await store.record(time.time())

    import redis.asyncio as aioredis

    client = aioredis.from_url(_redis_url(), decode_responses=True)
    ttl = await client.ttl(store._key)
    await client.aclose()
    assert ttl > 0


@pytest.mark.asyncio
async def test_earliest_returns_minimum_timestamp() -> None:
    store = _store()
    now = time.time()
    await store.record(now + 10)
    await store.record(now)
    await store.record(now + 5)

    earliest = await store.earliest()
    assert earliest is not None
    assert abs(earliest - now) < 1.0


@pytest.mark.asyncio
async def test_count_returns_zero_when_empty() -> None:
    store = _store()
    count = await store.count()
    assert count == 0
