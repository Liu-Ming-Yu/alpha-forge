"""Order intent and OMS lifecycle domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders.enums import (
    OrderSide,
    OrderStateEventType,
    OrderStatus,
    OrderType,
    TimeInForce,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime
    from decimal import Decimal


@dataclass(frozen=True)
class OrderIntent:
    """The system's desired trade action, independent of any broker specifics.

    Produced by the execution service when translating a PortfolioTarget into
    actionable trades.  Buy intents must pass a settled-cash eligibility check
    and receive a CashReservation before being promoted to a BrokerOrder.  Sell
    intents do not reserve cash, but still require position and risk approval.

    Args:
        order_id: Stable internal UUID.  This is the idempotency key that
            follows the order through its entire lifecycle.
        strategy_run_id: FK to the StrategyRun that generated this intent.
        portfolio_target_id: FK to the PortfolioTarget that drove this intent.
        instrument_id: FK to the instrument to trade.
        side: Buy or sell.
        quantity: Number of whole shares to trade (>= 1).
        order_type: Execution type requested.
        limit_price: Required if order_type is LIMIT or LOC.
        time_in_force: Order time-in-force instruction.
        created_at: UTC timestamp of intent creation.
        cash_reservation_id: UUID of the CashReservation created for this
            intent.  Must be set before buy orders can be submitted; remains
            None for sell orders.

    Must not contain:
        Broker-specific order IDs, exchange routing fields, or position-sizing
        logic.  Those belong in the execution adapter.
    """

    order_id: uuid.UUID
    strategy_run_id: uuid.UUID
    portfolio_target_id: uuid.UUID
    instrument_id: uuid.UUID
    side: OrderSide
    quantity: int
    order_type: OrderType
    time_in_force: TimeInForce
    created_at: datetime
    limit_price: Decimal | None = None
    cash_reservation_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")
        if self.order_type in (OrderType.LIMIT, OrderType.LOC) and self.limit_price is None:
            raise ValueError(f"limit_price required for order_type={self.order_type}")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be > 0")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class OrderStateEvent:
    """Append-only OMS event for one order lifecycle transition."""

    event_id: uuid.UUID
    order_id: uuid.UUID
    event_type: OrderStateEventType
    occurred_at: datetime
    status: OrderStatus | None = None
    broker_order_id: str | None = None
    idempotency_key: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        # Empty / whitespace-only idempotency keys mean "let me default it";
        # default to the event_id stringification so dedup checks remain
        # well-defined.  ``self.idempotency_key`` is annotated as ``str`` so
        # the only None case is callers that pass it explicitly via Any.
        if not (self.idempotency_key or "").strip():
            object.__setattr__(self, "idempotency_key", str(self.event_id))


@dataclass(frozen=True)
class CancelReplaceRequest:
    """EMS cancel/replace request for a live broker order."""

    request_id: uuid.UUID
    order_id: uuid.UUID
    broker_order_id: str
    requested_at: datetime
    new_limit_price: Decimal | None = None
    new_quantity: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.broker_order_id.strip():
            raise ValueError("broker_order_id must not be empty")
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        if self.new_limit_price is not None and self.new_limit_price <= 0:
            raise ValueError("new_limit_price must be > 0")
        if self.new_quantity is not None and self.new_quantity < 1:
            raise ValueError("new_quantity must be >= 1")
