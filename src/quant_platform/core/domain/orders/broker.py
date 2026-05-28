"""Broker order and fill domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.domain.orders.enums import OrderSide, OrderStatus


@dataclass(frozen=True)
class BrokerOrder:
    """An OrderIntent translated and submitted (or pending submission) to the broker.

    Carries both internal and broker identifiers so every fill can be
    reconciled back to its originating OrderIntent.

    Args:
        order_id: FK to OrderIntent.order_id (idempotency key).
        broker_order_id: Broker-assigned order ID.  None until the broker
            acknowledges the order.
        status: Current lifecycle status.
        submitted_at: UTC timestamp of submission.  None if not yet submitted.
        last_updated_at: UTC timestamp of most recent status change.
        filled_quantity: Cumulative shares filled so far.
        average_fill_price: Volume-weighted average fill price so far.
        rejection_reason: Broker-provided rejection message if status is
            REJECTED.

    Invariants:
        - filled_quantity <= original OrderIntent.quantity.
        - average_fill_price is None iff filled_quantity == 0.
        - broker_order_id must be set before status can move past SUBMITTED.
    """

    order_id: uuid.UUID
    status: OrderStatus
    last_updated_at: datetime
    filled_quantity: int = 0
    broker_order_id: str | None = None
    submitted_at: datetime | None = None
    average_fill_price: Decimal | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity must be >= 0")
        if self.filled_quantity > 0 and self.average_fill_price is None:
            raise ValueError("average_fill_price must be set when filled_quantity > 0")


@dataclass(frozen=True)
class FillEvent:
    """An immutable execution report from the broker.

    Fill events are append-only.  Broker corrections generate a new FillEvent
    with supersedes_id referencing the corrected record.

    Args:
        fill_id: Stable system UUID.
        order_id: FK to OrderIntent.order_id.
        broker_order_id: The broker's order identifier for correlation.
        broker_execution_id: Broker execution identifier for idempotency.
        instrument_id: FK to the instrument traded.
        side: Whether this fill was a buy or sell execution.
        quantity: Shares executed in this partial or full fill.
        fill_price: Execution price per share.
        commission: Total commission charged for this fill.
        currency: Currency of fill_price and commission.
        executed_at: UTC timestamp of execution reported by the broker.
        received_at: UTC timestamp when the system received this report.
        supersedes_id: If this corrects a prior fill, the UUID of that record.
    """

    fill_id: uuid.UUID
    order_id: uuid.UUID
    broker_order_id: str
    instrument_id: uuid.UUID
    side: OrderSide
    quantity: int
    fill_price: Decimal
    commission: Decimal
    currency: str
    executed_at: datetime
    received_at: datetime
    broker_execution_id: str | None = None
    supersedes_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError("fill quantity must be >= 1")
        if self.fill_price <= 0:
            raise ValueError("fill_price must be > 0")
        if self.commission < 0:
            raise ValueError("commission must be >= 0")
        if self.executed_at.tzinfo is None:
            raise ValueError("executed_at must be timezone-aware")
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must be timezone-aware")
        if self.broker_execution_id is not None and not self.broker_execution_id.strip():
            raise ValueError("broker_execution_id must not be blank when set")
