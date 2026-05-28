"""Position snapshot domain model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal


@dataclass(frozen=True)
class PositionSnapshot:
    """Point-in-time broker-reported position for one instrument.

    Args:
        snapshot_id: Stable system UUID for this snapshot record.
        instrument_id: FK to Instrument.
        quantity: Whole shares held (must be > 0).
        average_cost: Volume-weighted average cost basis per share.
        market_price: Last known market price per share.
        market_value: quantity × market_price.
        unrealised_pnl: market_value − (quantity × average_cost).
        as_of: UTC timestamp of the broker data this snapshot was built from.
        source: Either "broker" (reconciled directly) or "inferred"
            (computed from fills when broker data unavailable).

    Invariants:
        quantity > 0.  A position reaching zero quantity must be removed from
        the active position set, not stored as a zero-quantity snapshot.
    """

    snapshot_id: uuid.UUID
    instrument_id: uuid.UUID
    quantity: int
    average_cost: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealised_pnl: Decimal
    as_of: datetime
    source: str = "broker"

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.average_cost <= 0:
            raise ValueError("average_cost must be > 0")
        if self.market_price <= 0:
            raise ValueError("market_price must be > 0")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
