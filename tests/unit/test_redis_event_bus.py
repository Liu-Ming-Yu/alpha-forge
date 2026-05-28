"""Unit tests for RedisStreamsEventBus with a fake Redis client."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from datetime import UTC, datetime

import pytest

from quant_platform.core.events import OrderApproved
from quant_platform.infrastructure.event_bus import RedisStreamsEventBus


class _FakeRedisStreams:
    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._kv: dict[str, str] = {}
        self._groups: dict[tuple[str, str], str] = {}
        self._acked: set[tuple[str, str, str]] = set()
        self._pending: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = {}
        self.xadd_calls: list[dict[str, object]] = []

    @staticmethod
    def _id_num(stream_id: str) -> int:
        return int(stream_id.split("-", 1)[0])

    async def xadd(
        self,
        stream: str,
        fields: dict[str, str],
        *,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        self.xadd_calls.append(
            {
                "stream": stream,
                "maxlen": maxlen,
                "approximate": approximate,
            }
        )
        _ = approximate
        entries = self._streams.setdefault(stream, [])
        stream_id = f"{len(entries) + 1}-0"
        entries.append((stream_id, dict(fields)))
        if maxlen is not None and len(entries) > maxlen:
            self._streams[stream] = entries[-maxlen:]
        return stream_id

    async def xread(
        self,
        streams: dict[str, str],
        *,
        count: int = 1,
        block: int = 0,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        _ = block
        result: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
        for stream, last_id in streams.items():
            entries = self._streams.get(stream, [])
            items = [
                (sid, fields)
                for sid, fields in entries
                if self._id_num(sid) > self._id_num(last_id)
            ][:count]
            if items:
                result.append((stream, items))
        return result

    async def xrevrange(
        self,
        stream: str,
        *,
        count: int = 1,
    ) -> list[tuple[str, dict[str, str]]]:
        return list(reversed(self._streams.get(stream, [])))[:count]

    async def xgroup_create(
        self,
        stream: str,
        group: str,
        *,
        id: str = "$",
        mkstream: bool = False,
    ) -> bool:
        if mkstream:
            self._streams.setdefault(stream, [])
        key = (stream, group)
        if key in self._groups:
            raise RuntimeError("BUSYGROUP Consumer Group name already exists")
        if id == "$":
            entries = self._streams.get(stream, [])
            self._groups[key] = entries[-1][0] if entries else "0-0"
        else:
            self._groups[key] = id
        return True

    async def xreadgroup(
        self,
        *,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int = 1,
        block: int = 0,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        _ = consumername
        _ = block
        result: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
        for stream, requested in streams.items():
            key = (stream, groupname)
            if requested == "0":
                pending = [
                    item
                    for item in self._pending.get(key, [])
                    if (stream, groupname, item[0]) not in self._acked
                ][:count]
                if pending:
                    result.append((stream, pending))
                continue
            if requested != ">":
                continue
            last_id = self._groups.get(key, "0-0")
            entries = self._streams.get(stream, [])
            items = [
                (sid, fields)
                for sid, fields in entries
                if self._id_num(sid) > self._id_num(last_id)
            ][:count]
            if items:
                self._groups[key] = items[-1][0]
                pending = self._pending.setdefault(key, [])
                for item in items:
                    if item[0] not in {existing[0] for existing in pending}:
                        pending.append(item)
                result.append((stream, items))
        return result

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        self._acked.add((stream, group, message_id))
        key = (stream, group)
        self._pending[key] = [item for item in self._pending.get(key, []) if item[0] != message_id]
        return 1

    async def get(self, key: str) -> str | None:
        return self._kv.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        _ = ex
        if nx and key in self._kv:
            return False
        self._kv[key] = value
        return True

    async def delete(self, key: str) -> int:
        existed = key in self._kv
        self._kv.pop(key, None)
        return 1 if existed else 0


class _FailOnceRedisStreams(_FakeRedisStreams):
    def __init__(self) -> None:
        super().__init__()
        self._fail_next_xadd = True

    async def xadd(
        self,
        stream: str,
        fields: dict[str, str],
        *,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        if self._fail_next_xadd:
            self._fail_next_xadd = False
            raise RuntimeError("transient xadd failure")
        return await super().xadd(
            stream,
            fields,
            maxlen=maxlen,
            approximate=approximate,
        )


@pytest.mark.asyncio
async def test_publish_and_recent_events_round_trip() -> None:
    fake = _FakeRedisStreams()
    bus = RedisStreamsEventBus("redis://fake")
    bus._client = fake  # type: ignore[attr-defined]
    now = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )

    await bus.publish(event)
    history = await bus.recent_events(limit=10, event_type=OrderApproved)
    assert len(history) == 1
    assert history[0].order_id == event.order_id
    assert fake.xadd_calls[0]["maxlen"] is None


@pytest.mark.asyncio
async def test_publish_uses_maxlen_only_without_consumer_groups() -> None:
    fake = _FakeRedisStreams()
    bus = RedisStreamsEventBus("redis://fake", maxlen=5, use_consumer_groups=False)
    bus._client = fake  # type: ignore[attr-defined]
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=datetime(2026, 1, 5, 14, 0, tzinfo=UTC),
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )

    await bus.publish(event)

    assert fake.xadd_calls[0]["maxlen"] == 5


@pytest.mark.asyncio
async def test_publish_dedupe_skips_duplicate_event_id() -> None:
    fake = _FakeRedisStreams()
    bus = RedisStreamsEventBus("redis://fake")
    bus._client = fake  # type: ignore[attr-defined]
    now = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )
    await bus.publish(event)
    await bus.publish(event)
    history = await bus.recent_events(limit=10, event_type=OrderApproved)
    assert len(history) == 1


@pytest.mark.asyncio
async def test_failed_xadd_preserves_dedupe_guard() -> None:
    """After a failed xadd, the dedupe guard must NOT be cleared.

    Clearing the guard on failure would allow a duplicate on the next retry —
    dangerous if the xadd actually succeeded but we didn't receive the ack.
    The operator must explicitly clear the guard (force_retry) to re-publish.
    """
    fake = _FailOnceRedisStreams()
    bus = RedisStreamsEventBus("redis://fake")
    bus._client = fake  # type: ignore[attr-defined]
    now = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )

    with pytest.raises(RuntimeError, match="transient xadd failure"):
        await bus.publish(event)

    # Retry without clearing the guard: dedupe guard is still set from the
    # failed attempt, so this call is silently skipped (no error, no publish).
    await bus.publish(event)
    history = await bus.recent_events(limit=10, event_type=OrderApproved)
    # The event was not published on either attempt due to xadd failure +
    # subsequent deduplication guard blocking the retry.
    assert len(history) == 0


@pytest.mark.asyncio
async def test_subscribe_reads_events_and_tracks_offset() -> None:
    fake = _FakeRedisStreams()
    bus = RedisStreamsEventBus("redis://fake")
    bus._client = fake  # type: ignore[attr-defined]
    now = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    first = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )
    second = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )
    await bus.publish(first)
    consumer = bus.subscribe(OrderApproved, "consumer-a")
    got_first = await asyncio.wait_for(anext(consumer), timeout=0.2)
    assert got_first.order_id == first.order_id
    next_read = asyncio.create_task(anext(consumer))
    await asyncio.sleep(0.01)
    next_read.cancel()
    with suppress(asyncio.CancelledError):
        await next_read
    await consumer.aclose()

    await bus.publish(second)
    consumer_2 = bus.subscribe(OrderApproved, "consumer-a")
    got_second = await asyncio.wait_for(anext(consumer_2), timeout=0.2)
    assert got_second.order_id == second.order_id
    await consumer_2.aclose()


@pytest.mark.asyncio
async def test_consumer_group_redelivery_reprocesses_after_failed_handler() -> None:
    fake = _FakeRedisStreams()
    bus = RedisStreamsEventBus("redis://fake")
    bus._client = fake  # type: ignore[attr-defined]
    now = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )
    await bus.publish(event)

    consumer = bus.subscribe(OrderApproved, "consumer-a")
    got = await asyncio.wait_for(anext(consumer), timeout=0.2)
    assert got.event_id == event.event_id
    with pytest.raises(RuntimeError, match="handler failed"):
        await consumer.athrow(RuntimeError("handler failed"))

    stream = bus._stream_name(OrderApproved)  # type: ignore[attr-defined]
    group = bus._group_name(OrderApproved, "consumer-a")  # type: ignore[attr-defined]
    assert fake._acked == set()
    assert await fake.get(f"{stream}:consumed:{group}:{event.event_id}") is None

    retry_consumer = bus.subscribe(OrderApproved, "consumer-a")
    retried = await asyncio.wait_for(anext(retry_consumer), timeout=0.2)
    assert retried.event_id == event.event_id
    next_read = asyncio.create_task(anext(retry_consumer))
    await asyncio.sleep(0.01)
    next_read.cancel()
    with suppress(asyncio.CancelledError):
        await next_read
    assert (stream, group, "1-0") in fake._acked
    assert await fake.get(f"{stream}:consumed:{group}:{event.event_id}") == "1"
    await retry_consumer.aclose()
