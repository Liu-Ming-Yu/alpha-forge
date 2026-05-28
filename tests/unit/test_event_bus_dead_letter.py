"""Tests for the Redis Streams dead-letter sweep (Phase 4.3)."""

from __future__ import annotations

import pytest

from quant_platform.infrastructure.event_bus import RedisStreamsEventBus


class _FakeRedis:
    """Minimal ``XPENDING``/``XADD``/``XACK``-compatible fake."""

    def __init__(self) -> None:
        self.groups = [{"name": "g1"}]
        self.pending_entries: list[dict[str, object]] = []
        self.stream_rows: dict[str, dict[str, str]] = {}
        self.xadds: list[tuple[str, dict[str, str]]] = []
        self.xacks: list[tuple[str, str, str]] = []

    async def xinfo_groups(self, stream: str) -> list[dict[str, object]]:
        return self.groups

    async def xpending_range(
        self, *, name: str, groupname: str, min: str, max: str, count: int
    ) -> list[dict[str, object]]:
        return self.pending_entries

    async def xrange(
        self, stream: str, *, min: str, max: str, count: int
    ) -> list[tuple[str, dict[str, str]]]:
        row = self.stream_rows.get(min)
        return [(min, row)] if row else []

    async def xadd(self, stream: str, fields: dict[str, str]) -> str:
        self.xadds.append((stream, dict(fields)))
        return "1-0"

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        self.xacks.append((stream, group, message_id))
        return 1


@pytest.mark.asyncio
async def test_sweep_moves_over_retried_entries() -> None:
    bus = RedisStreamsEventBus(
        redis_url="redis://x",
        dead_letter_after_retries=3,
    )
    fake = _FakeRedis()
    fake.pending_entries = [
        {"message_id": "1-0", "times_delivered": 5},
        {"message_id": "2-0", "times_delivered": 1},
    ]
    fake.stream_rows = {
        "1-0": {"payload": "p1", "event_id": "e1"},
        "2-0": {"payload": "p2", "event_id": "e2"},
    }
    bus._client = fake  # noqa: SLF001 - test injection

    moved = await bus.sweep_dead_letters("qp:events:FillReceived")
    assert moved == 1
    assert fake.xadds == [
        (
            "qp:events:FillReceived.dlq",
            {
                "payload": "p1",
                "event_id": "e1",
                "orig_stream": "qp:events:FillReceived",
                "orig_group": "g1",
                "orig_id": "1-0",
                "delivery_count": "5",
            },
        )
    ]
    assert fake.xacks == [("qp:events:FillReceived", "g1", "1-0")]


@pytest.mark.asyncio
async def test_sweep_noop_when_budget_zero() -> None:
    bus = RedisStreamsEventBus(redis_url="redis://x", dead_letter_after_retries=0)
    bus._client = _FakeRedis()  # noqa: SLF001
    assert await bus.sweep_dead_letters("stream") == 0


# ---------------------------------------------------------------------------
# R-DAT-05: subscribe() must ack only *after* the consumer handles the event.
# ---------------------------------------------------------------------------


import uuid
from datetime import UTC, datetime

from quant_platform.core.events import OrderSubmitted
from quant_platform.infrastructure.event_bus import _serialize_event


class _SubscribeFake:
    """Minimal consumer-group fake capturing ack ordering vs yield."""

    def __init__(self, payload: str) -> None:
        self._queue: list[tuple[str, str, dict[str, str]]] = [
            ("qp:events:OrderSubmitted", "1-0", {"payload": payload, "event_id": "e1"}),
        ]
        self.xacks: list[str] = []
        self.xgroup_calls: list[tuple[str, str]] = []

    async def xgroup_create(
        self, stream: str, group: str, id: str = "$", mkstream: bool = False
    ) -> None:
        self.xgroup_calls.append((stream, group))

    async def xreadgroup(
        self,
        *,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int,
        block: int,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        if self._queue:
            stream = next(iter(streams))
            msg_stream, mid, fields = self._queue.pop(0)
            return [(msg_stream, [(mid, fields)])]
        return []

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        self.xacks.append(message_id)
        return 1

    async def set(self, *args: object, **kwargs: object) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return None


def _submitted_event() -> OrderSubmitted:
    return OrderSubmitted(
        event_id=uuid.uuid4(),
        occurred_at=datetime.now(tz=UTC),
        order_id=uuid.uuid4(),
        broker_order_id="BRK-1",
    )


import asyncio


@pytest.mark.asyncio
async def test_successful_handler_acks_exactly_once() -> None:
    """Consumer processes event normally → ack happens exactly once.

    The subscribe() generator yields the event, then — once the consumer
    returns control via the next iteration — acks.  We drive this with a
    short ``wait_for`` that deliberately times out after the ack branch
    has executed but before a (non-existent) next event would be
    produced.
    """
    event = _submitted_event()
    fake = _SubscribeFake(payload=_serialize_event(event))
    bus = RedisStreamsEventBus(redis_url="redis://x", block_ms=0, use_consumer_groups=True)
    bus._client = fake  # noqa: SLF001

    agen = bus.subscribe(OrderSubmitted, consumer_id="c1")
    received = await agen.__anext__()
    assert received.order_id == event.order_id
    # Ack must NOT have happened yet — the consumer is still "handling".
    assert fake.xacks == []
    # Resume the generator so the post-yield ack runs, then let it idle.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agen.__anext__(), timeout=0.1)
    assert fake.xacks == ["1-0"], "ack must fire after handler returns successfully"
    await agen.aclose()


@pytest.mark.asyncio
async def test_failed_handler_leaves_entry_in_pel() -> None:
    """Consumer raises → ack must NOT fire, entry stays in PEL for DLQ."""
    event = _submitted_event()
    fake = _SubscribeFake(payload=_serialize_event(event))
    bus = RedisStreamsEventBus(redis_url="redis://x", block_ms=0, use_consumer_groups=True)
    bus._client = fake  # noqa: SLF001

    agen = bus.subscribe(OrderSubmitted, consumer_id="c2")
    received = await agen.__anext__()
    assert received.order_id == event.order_id

    # Simulate consumer crash mid-handle by throwing into the generator.
    class _HandlerError(Exception):
        pass

    with pytest.raises(_HandlerError):
        await agen.athrow(_HandlerError("simulated subscriber crash"))

    # Crucially: the message remains un-acked so sweep_dead_letters can
    # eventually move it to the DLQ.
    assert fake.xacks == []
