"""Tests for IB historical-data pacing state durability (Phase 3.3).

Focuses on ``InMemoryPacingStore`` contract compliance; the Redis
implementation mirrors this surface and is covered by the integration
suite.
"""

from __future__ import annotations

import pytest

from quant_platform.services.execution_service.stores.pacing_store import (
    InMemoryPacingStore,
    build_pacing_store,
)


@pytest.mark.asyncio
async def test_in_memory_store_basic_round_trip() -> None:
    store = InMemoryPacingStore()
    assert await store.count() == 0
    assert await store.earliest() is None

    await store.record(100.0)
    await store.record(105.0)
    await store.record(150.0)

    assert await store.count() == 3
    assert await store.earliest() == 100.0
    assert await store.hydrate() == [100.0, 105.0, 150.0]


@pytest.mark.asyncio
async def test_in_memory_store_prune_before_drops_stale_entries() -> None:
    store = InMemoryPacingStore()
    for t in (100.0, 105.0, 150.0, 200.0):
        await store.record(t)

    await store.prune_before(120.0)
    assert await store.hydrate() == [150.0, 200.0]


def test_build_pacing_store_falls_back_when_no_redis_url() -> None:
    store = build_pacing_store(redis_url=None, client_id=42, window_seconds=600.0)
    assert isinstance(store, InMemoryPacingStore)


def test_build_pacing_store_selects_redis_when_url_given() -> None:
    from quant_platform.services.execution_service.stores.pacing_store import (
        RedisPacingStore,
    )

    store = build_pacing_store(
        redis_url="redis://localhost:6379/0",
        client_id=17,
        window_seconds=600.0,
    )
    assert isinstance(store, RedisPacingStore)
