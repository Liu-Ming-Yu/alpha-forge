"""Execution route and quality domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.orders.enums import ExecutionTactic


@dataclass(frozen=True)
class VenueRoute:
    """Venue/routing instruction emitted by the EMS before broker translation."""

    route_id: uuid.UUID
    venue: str
    tactic: ExecutionTactic
    max_participation_rate: Decimal
    urgency: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not self.venue.strip():
            raise ValueError("venue must not be empty")
        if not (Decimal("0") <= self.max_participation_rate <= Decimal("1")):
            raise ValueError("max_participation_rate must be in [0, 1]")
        if not (Decimal("0") <= self.urgency <= Decimal("1")):
            raise ValueError("urgency must be in [0, 1]")


@dataclass(frozen=True)
class ExecutionQualityReport:
    """Post-trade execution quality report for one order."""

    report_id: uuid.UUID
    order_id: uuid.UUID
    as_of: datetime
    venue: str
    tactic: ExecutionTactic
    arrival_price: Decimal | None = None
    decision_price: Decimal | None = None
    vwap: Decimal | None = None
    fill_price: Decimal | None = None
    slippage_bps: Decimal | None = None
    participation_rate: Decimal | None = None
    passed: bool = True

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.venue.strip():
            raise ValueError("venue must not be empty")
        if self.participation_rate is not None and not (
            Decimal("0") <= self.participation_rate <= Decimal("1")
        ):
            raise ValueError("participation_rate must be in [0, 1]")
