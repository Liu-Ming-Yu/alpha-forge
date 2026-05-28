"""In-memory V2 order-state repository."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.orders import OrderStateEvent


class InMemoryOrderStateStore:
    """Idempotent append-only OMS event store."""

    def __init__(self) -> None:
        self._events: dict[uuid.UUID, list[OrderStateEvent]] = defaultdict(list)
        self._idempotency_keys: set[str] = set()

    async def append(self, event: OrderStateEvent) -> None:
        if event.idempotency_key in self._idempotency_keys:
            return
        self._idempotency_keys.add(event.idempotency_key)
        rows = self._events[event.order_id]
        rows.append(event)
        rows.sort(key=lambda item: item.occurred_at)

    async def list_events(self, order_id: uuid.UUID) -> list[OrderStateEvent]:
        return list(self._events.get(order_id, []))

    async def latest(self, order_id: uuid.UUID) -> OrderStateEvent | None:
        rows = self._events.get(order_id, [])
        return rows[-1] if rows else None
