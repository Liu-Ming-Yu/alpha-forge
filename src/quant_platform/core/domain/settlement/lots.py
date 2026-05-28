"""Settlement-lot domain model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime


class SettlementStatus(StrEnum):
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass(frozen=True)
class SettlementLot:
    """A projected or confirmed cash settlement from a sale.

    Created when a sell FillEvent is received.  The settlement_date is
    calculated by the SettlementCalendar using the trade_date and the
    standard T+1 rule for US equities (post-2024 settlement reform).

    Args:
        lot_id: Stable system UUID.
        fill_id: FK to the FillEvent that created this lot.
        instrument_id: FK to the instrument sold.
        trade_date: The date on which the sale was executed.
        settlement_date: The date on which the proceeds are expected to settle.
        gross_proceeds: Total sale proceeds before commission (quantity × price).
        commission: Commission deducted from proceeds.
        net_proceeds: gross_proceeds − commission.  This is the amount that
            will become settled_cash on settlement_date.
        currency: ISO currency code.
        status: Settlement lifecycle status.
        settled_at: UTC timestamp of actual settlement confirmation.
            None until the lot is confirmed settled.
    """

    lot_id: uuid.UUID
    fill_id: uuid.UUID
    instrument_id: uuid.UUID
    trade_date: date
    settlement_date: date
    gross_proceeds: Decimal
    commission: Decimal
    net_proceeds: Decimal
    currency: str
    status: SettlementStatus
    settled_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.net_proceeds <= 0:
            raise ValueError("net_proceeds must be > 0")
        if self.settlement_date < self.trade_date:
            raise ValueError("settlement_date must be >= trade_date")
        expected_net = self.gross_proceeds - self.commission
        if abs(self.net_proceeds - expected_net) > Decimal("0.001"):
            raise ValueError(
                f"net_proceeds {self.net_proceeds} != gross_proceeds - commission {expected_net}"
            )
