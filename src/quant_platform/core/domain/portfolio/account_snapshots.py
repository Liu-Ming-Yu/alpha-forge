"""Account snapshot domain model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.portfolio.position_snapshots import PositionSnapshot


@dataclass(frozen=True)
class AccountSnapshot:
    """Point-in-time view of the cash-account balance and exposure.

    Produced by the execution service after reconciling with the broker.
    Consumed by CashConstraintEngine and PortfolioConstructor.

    Args:
        snapshot_id: Stable system UUID.
        as_of: UTC timestamp of the broker data this snapshot was built from.
        settled_cash: Fully settled cash available for trading.
        unsettled_cash: Proceeds from sales awaiting settlement (T+1/T+2).
            This is for display/reporting only; the CashLedger tracks the
            authoritative unsettled pool for gating purposes.
        reserved_cash: Cash reserved by pending OrderIntents.
            reserved_cash <= settled_cash always.
        available_cash: settled_cash − reserved_cash.
        net_asset_value: settled_cash + unsettled_cash + sum(market_values).
        positions: All open positions at this point in time.
        source: "broker" (reconciled from broker data) or "inferred"
            (derived from fills when broker data was unavailable).

    Invariants:
        settled_cash >= 0.
        reserved_cash <= settled_cash.
        available_cash == settled_cash − reserved_cash.
    """

    snapshot_id: uuid.UUID
    as_of: datetime
    settled_cash: Decimal
    unsettled_cash: Decimal
    reserved_cash: Decimal
    available_cash: Decimal
    net_asset_value: Decimal
    positions: tuple[PositionSnapshot, ...]
    source: str = "broker"

    def __post_init__(self) -> None:
        if self.settled_cash < 0:
            raise ValueError("settled_cash must be >= 0 for a cash account")
        if self.unsettled_cash < 0:
            raise ValueError("unsettled_cash must be >= 0")
        if self.reserved_cash < 0:
            raise ValueError("reserved_cash must be >= 0")
        if self.reserved_cash > self.settled_cash:
            raise ValueError("reserved_cash must be <= settled_cash")
        expected_available = self.settled_cash - self.reserved_cash
        if abs(self.available_cash - expected_available) > Decimal("0.01"):
            raise ValueError(
                f"available_cash {self.available_cash} != "
                f"settled_cash - reserved_cash {expected_available}"
            )
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
