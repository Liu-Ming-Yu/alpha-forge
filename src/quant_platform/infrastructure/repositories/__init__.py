"""In-memory repository implementations for testing and paper trading.

These adapters satisfy the OrderRepository and PositionRepository contracts
without any database dependency.  They are suitable for unit tests, backtest
engines, and early-stage paper trading before PostgreSQL is integrated.

Thread safety: asyncio.Lock guards all shared state, making these adapters
safe under concurrent async tasks (e.g. event-bus consumer + reconciliation
running in the same event loop).  They remain single-process only — for
multi-process deployments switch to the PostgreSQL-backed implementations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.orders import FillEvent, OrderIntent
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


class InMemoryOrderRepository:
    """In-memory OrderRepository for testing and paper trading."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._intents: dict[uuid.UUID, OrderIntent] = {}
        self._fills: dict[uuid.UUID, list[FillEvent]] = {}
        self._terminal_order_ids: set[uuid.UUID] = set()
        self._terminal_reasons: dict[uuid.UUID, str] = {}

    async def save_intent(self, intent: OrderIntent) -> None:
        async with self._lock:
            if intent.order_id in self._intents:
                raise ValueError(f"Duplicate order_id: {intent.order_id}")
            self._intents[intent.order_id] = intent

    async def get_intent(self, order_id: uuid.UUID) -> OrderIntent | None:
        async with self._lock:
            return self._intents.get(order_id)

    async def save_fill(self, fill: FillEvent) -> None:
        async with self._lock:
            fills = self._fills.setdefault(fill.order_id, [])
            if any(f.fill_id == fill.fill_id for f in fills):
                return
            if fill.broker_execution_id is not None and any(
                f.broker_order_id == fill.broker_order_id
                and f.broker_execution_id == fill.broker_execution_id
                for f in fills
            ):
                return
            fills.append(fill)
            self._maybe_mark_terminal_from_fills_locked(fill.order_id)

    async def get_fills(self, order_id: uuid.UUID) -> list[FillEvent]:
        async with self._lock:
            return list(self._fills.get(order_id, []))

    async def list_open_orders(self, strategy_run_id: uuid.UUID) -> list[OrderIntent]:
        async with self._lock:
            return [
                i
                for i in self._intents.values()
                if i.strategy_run_id == strategy_run_id
                and i.order_id not in self._terminal_order_ids
            ]

    async def mark_terminal(self, order_id: uuid.UUID, reason: str) -> None:
        """Mark an order as terminal so it no longer appears in open-order views."""
        async with self._lock:
            if order_id not in self._intents:
                return
            self._terminal_order_ids.add(order_id)
            self._terminal_reasons[order_id] = reason

    async def is_terminal(self, order_id: uuid.UUID) -> bool:
        async with self._lock:
            return order_id in self._terminal_order_ids

    async def terminal_reason(self, order_id: uuid.UUID) -> str | None:
        async with self._lock:
            return self._terminal_reasons.get(order_id)

    def _maybe_mark_terminal_from_fills_locked(self, order_id: uuid.UUID) -> None:
        """Must be called while self._lock is held."""
        intent = self._intents.get(order_id)
        if intent is None:
            return
        filled_qty = sum(fill.quantity for fill in self._fills.get(order_id, []))
        if filled_qty >= intent.quantity:
            self._terminal_order_ids.add(order_id)
            self._terminal_reasons.setdefault(order_id, "fully filled")


class InMemoryPositionRepository:
    """In-memory PositionRepository for testing and paper trading."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._snapshots: list[AccountSnapshot] = []

    async def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        async with self._lock:
            self._snapshots.append(snapshot)

    async def get_latest_snapshot(self) -> AccountSnapshot | None:
        async with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    async def get_snapshot_at(self, as_of: datetime) -> AccountSnapshot | None:
        async with self._lock:
            candidates = [s for s in self._snapshots if s.as_of <= as_of]
            return candidates[-1] if candidates else None
