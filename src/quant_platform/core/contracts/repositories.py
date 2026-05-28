"""Repository contracts: orders and positions persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.orders import FillEvent, OrderIntent
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


@runtime_checkable
class OrderRepository(Protocol):
    """Persistence interface for the order lifecycle.

    Must never:
        Allow direct mutation of a stored OrderIntent or FillEvent.
        Return a different order when the same order_id is queried twice
        within one session (consistency guarantee).

    Research-to-production parity requirement:
        The same interface is used in backtest (in-memory adapter) and
        live (PostgreSQL adapter).
    """

    async def save_intent(self, intent: OrderIntent) -> None:
        """Persist a new OrderIntent.  Raises on duplicate order_id."""
        ...

    async def get_intent(self, order_id: uuid.UUID) -> OrderIntent | None:
        """Return the OrderIntent for the given order_id, or None."""
        ...

    async def save_fill(self, fill: FillEvent) -> None:
        """Persist a FillEvent.  Idempotent on fill_id."""
        ...

    async def get_fills(self, order_id: uuid.UUID) -> list[FillEvent]:
        """Return all fills for the given order_id, oldest first."""
        ...

    async def list_open_orders(self, strategy_run_id: uuid.UUID) -> list[OrderIntent]:
        """Return all non-terminal OrderIntents for the given run."""
        ...


@runtime_checkable
class PositionRepository(Protocol):
    """Persistence interface for position and account snapshots.

    Must never:
        Store a PositionSnapshot with quantity == 0.
        Return stale snapshots as current without flagging the staleness.
    """

    async def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        """Persist an AccountSnapshot and all its PositionSnapshots."""
        ...

    async def get_latest_snapshot(self) -> AccountSnapshot | None:
        """Return the most recently stored AccountSnapshot, or None."""
        ...

    async def get_snapshot_at(self, as_of: datetime) -> AccountSnapshot | None:
        """Return the AccountSnapshot nearest to but not after as_of."""
        ...
