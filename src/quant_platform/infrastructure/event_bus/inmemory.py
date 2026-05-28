"""In-memory EventBus adapter."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from quant_platform.infrastructure.metrics import record_event_publish

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from quant_platform.core.events import DomainEvent


class InMemoryEventBus:
    """In-memory at-most-once event bus for single-process deployments."""

    def __init__(self, *, max_history: int = 10_000) -> None:
        self._queues: dict[type[DomainEvent], dict[str, asyncio.Queue[DomainEvent]]] = defaultdict(
            dict
        )
        self._history: deque[DomainEvent] = deque(maxlen=max_history)

    async def publish(self, event: DomainEvent) -> None:
        self._history.append(event)
        event_type = type(event)
        for q in self._queues.get(event_type, {}).values():
            await q.put(event)
        record_event_publish(backend="in_memory", outcome="ok")

    async def subscribe(
        self,
        event_type: type[DomainEvent],
        consumer_id: str,
    ) -> AsyncIterator[DomainEvent]:
        q: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._queues[event_type][consumer_id] = q
        try:
            while True:
                yield await q.get()
        finally:
            self._queues[event_type].pop(consumer_id, None)

    @property
    def history(self) -> list[DomainEvent]:
        return list(self._history)

    async def recent_events(
        self,
        *,
        limit: int = 1000,
        event_type: type[DomainEvent] | None = None,
    ) -> list[DomainEvent]:
        h = list(self._history)
        if event_type is None:
            return h[-limit:]
        filtered = [e for e in h if isinstance(e, event_type)]
        return filtered[-limit:]
