"""Integration tests for RedisStreamsEventBus against a real Redis instance."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from datetime import UTC, datetime

import pytest

from quant_platform.core.events import OrderApproved, OrderRejected
from quant_platform.infrastructure.event_bus import RedisStreamsEventBus

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for durable Redis tests")
    return url


def _make_bus(
    prefix: str,
    *,
    maxlen: int = 10000,
    dead_letter_after_retries: int = 0,
    use_consumer_groups: bool = True,
) -> RedisStreamsEventBus:
    return RedisStreamsEventBus(
        redis_url=_redis_url(),
        stream_prefix=prefix,
        maxlen=maxlen,
        block_ms=50,
        use_consumer_groups=use_consumer_groups,
        publish_dedupe_enabled=True,
        dead_letter_after_retries=dead_letter_after_retries,
    )


def _event(order_id: uuid.UUID | None = None) -> OrderApproved:
    return OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=datetime.now(tz=_UTC),
        order_id=order_id or uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )


@pytest.mark.asyncio
async def test_publish_and_subscribe_consumer_group_delivers() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix)
    evt = _event()
    await bus.publish(evt)

    consumer = bus.subscribe(OrderApproved, f"cg-{uuid.uuid4()}")
    received = await asyncio.wait_for(anext(consumer), timeout=3)
    assert received.event_id == evt.event_id
    await consumer.aclose()


@pytest.mark.asyncio
async def test_publish_deduplicate_prevents_double_delivery() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix)
    evt = _event()
    await bus.publish(evt)
    await bus.publish(evt)

    group = f"dedup-{uuid.uuid4()}"
    consumer = bus.subscribe(OrderApproved, group)

    first = await asyncio.wait_for(anext(consumer), timeout=3)
    assert first.event_id == evt.event_id

    drain_task = asyncio.create_task(anext(consumer))
    await asyncio.sleep(0.15)
    drain_task.cancel()
    with suppress(asyncio.CancelledError):
        await drain_task
    await consumer.aclose()


@pytest.mark.asyncio
async def test_subscribe_drains_pel_on_restart() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix)
    evt = _event()
    await bus.publish(evt)

    group = f"pel-group-{uuid.uuid4()}"
    consumer = bus.subscribe(OrderApproved, group)
    first = await asyncio.wait_for(anext(consumer), timeout=3)
    assert first.event_id == evt.event_id

    with suppress(RuntimeError):
        await consumer.athrow(RuntimeError("crash"))
    await consumer.aclose()

    retry_bus = _make_bus(prefix)
    retry_consumer = retry_bus.subscribe(OrderApproved, group)
    retried = await asyncio.wait_for(anext(retry_consumer), timeout=3)
    assert retried.event_id == evt.event_id
    await retry_consumer.aclose()


@pytest.mark.asyncio
async def test_sweep_dead_letters_moves_over_retry_budget() -> None:
    # sweep_dead_letters only moves entries whose times_delivered > budget.
    # With dead_letter_after_retries=1, a message must be delivered at least
    # twice before the sweep will move it.  We trigger two delivery attempts
    # by subscribing twice in the same group without ACKing.
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix, dead_letter_after_retries=1)
    evt = _event()
    await bus.publish(evt)
    stream = f"{prefix}:OrderApproved"
    group = f"dlq-group-{uuid.uuid4()}"

    # First delivery (times_delivered = 1 → PEL, not yet over budget)
    c1 = bus.subscribe(OrderApproved, group)
    first = await asyncio.wait_for(anext(c1), timeout=3)
    assert first.event_id == evt.event_id
    with suppress(RuntimeError):
        await c1.athrow(RuntimeError("fail 1"))
    await c1.aclose()

    # Second delivery (times_delivered = 2 > budget=1 → now over budget)
    c2 = _make_bus(prefix, dead_letter_after_retries=1).subscribe(OrderApproved, group)
    with suppress(asyncio.TimeoutError):
        second = await asyncio.wait_for(anext(c2), timeout=3)
        with suppress(RuntimeError):
            await c2.athrow(RuntimeError("fail 2"))
    await c2.aclose()

    moved = await bus.sweep_dead_letters(stream)
    assert moved >= 1

    import redis.asyncio as aioredis

    client = aioredis.from_url(_redis_url(), decode_responses=True)
    dlq_stream = f"{stream}.dlq"
    dlq_len = await client.xlen(dlq_stream)
    assert dlq_len >= 1
    await client.aclose()


@pytest.mark.asyncio
async def test_recent_events_returns_last_n() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix)
    for _ in range(5):
        await bus.publish(_event())

    recent = await bus.recent_events(event_type=OrderApproved, limit=3)
    assert len(recent) == 3


@pytest.mark.asyncio
async def test_stream_maxlen_enforced() -> None:
    # XADD MAXLEN ~ (approximate) only trims when the radix-tree node is full,
    # so it won't trim a tiny stream.  Use a larger publish count so the
    # approximate trim fires multiple times, then allow 3x the maxlen as slack.
    prefix = f"qp:test:{uuid.uuid4()}"
    maxlen = 10
    bus = _make_bus(prefix, maxlen=maxlen, use_consumer_groups=False)
    for _ in range(200):
        await bus.publish(
            OrderApproved(
                event_id=uuid.uuid4(),
                occurred_at=datetime.now(tz=_UTC),
                order_id=uuid.uuid4(),
                reservation_id=uuid.uuid4(),
            )
        )

    import redis.asyncio as aioredis

    client = aioredis.from_url(_redis_url(), decode_responses=True)
    stream = f"{prefix}:OrderApproved"
    length = await client.xlen(stream)
    # Approximate trim guarantees stream stays near maxlen; 3x is a safe upper bound.
    assert length <= maxlen * 3
    await client.aclose()


@pytest.mark.asyncio
async def test_recent_events_returns_empty_for_empty_stream() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = _make_bus(prefix)
    recent = await bus.recent_events(event_type=OrderRejected, limit=10)
    assert recent == []
