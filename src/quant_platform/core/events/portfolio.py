"""Portfolio-construction domain events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from quant_platform.core.events.base import DomainEvent


@dataclass(frozen=True)
class PortfolioTargetBuilt(DomainEvent):
    """A new PortfolioTarget has been constructed and is ready for execution.

    Args:
        target_id: FK to PortfolioTarget.
        strategy_run_id: FK to the StrategyRun.
        regime_id: FK to the RegimeState used in construction.
    """

    target_id: uuid.UUID
    strategy_run_id: uuid.UUID
    regime_id: uuid.UUID


@dataclass(frozen=True)
class OrderApproved(DomainEvent):
    """An OrderIntent has passed all risk and cash-gate checks.

    Args:
        order_id: FK to OrderIntent.
        reservation_id: FK to the CashReservation created for this order.
            Buy orders must have a reservation; sell orders do not reserve
            cash and therefore publish this as None.
    """

    order_id: uuid.UUID
    reservation_id: uuid.UUID | None


@dataclass(frozen=True)
class OrderRejected(DomainEvent):
    """An OrderIntent has been rejected by a risk or cash-gate check.

    Args:
        order_id: FK to OrderIntent.
        reason: Human-readable rejection reason.
    """

    order_id: uuid.UUID
    reason: str
