"""IB order lifecycle mapping helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypedDict

from quant_platform.core.domain.orders import BrokerOrder, FillEvent, OrderSide, OrderStatus
from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerLifecycleEvent,
    BrokerOrderCancelled,
    BrokerOrderCompleted,
    BrokerOrderRejected,
    BrokerUnmatchedFill,
)

IB_CANCELLED_STATUSES: frozenset[str] = frozenset({"Cancelled", "ApiCancelled"})


class PendingExecution(TypedDict):
    """Execution details held until IB emits the matching commission report."""

    internal_id: uuid.UUID | None
    ib_order_id: int
    exec_id: str
    contract: Any
    shares: int
    price: Decimal
    side: str
    time: datetime
    cum_qty: Decimal


def broker_order_status_from_ib(status: str) -> OrderStatus:
    """Map an IB open-order status string to the platform order status enum."""
    status_map = {
        "Submitted": OrderStatus.SUBMITTED,
        "PreSubmitted": OrderStatus.SUBMITTED,
        "Filled": OrderStatus.FILLED,
        "Cancelled": OrderStatus.CANCELLED,
        "ApiCancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.REJECTED,
    }
    return status_map.get(status, OrderStatus.SUBMITTED)


def broker_order_from_open_order(
    *,
    order_ref: str,
    status: str,
    broker_order_id: str,
    observed_at: datetime,
) -> BrokerOrder:
    """Build the platform open-order projection from an IB openOrder callback."""
    order_id = uuid.UUID(order_ref) if order_ref else uuid.uuid4()
    return BrokerOrder(
        order_id=order_id,
        status=broker_order_status_from_ib(status),
        last_updated_at=observed_at,
        broker_order_id=broker_order_id,
        filled_quantity=0,
    )


def order_lifecycle_event_from_status(
    *,
    order_id: uuid.UUID,
    broker_order_id: str,
    status: str,
    remaining: Decimal,
    occurred_at: datetime,
) -> BrokerLifecycleEvent | None:
    """Build a terminal broker lifecycle event from an IB orderStatus callback."""
    if status == "Filled" and Decimal(str(remaining)) == Decimal("0"):
        return BrokerOrderCompleted(
            order_id=order_id,
            broker_order_id=broker_order_id,
            occurred_at=occurred_at,
        )
    if status in IB_CANCELLED_STATUSES:
        return BrokerOrderCancelled(
            order_id=order_id,
            broker_order_id=broker_order_id,
            reason="broker cancelled",
            occurred_at=occurred_at,
        )
    if status == "Inactive":
        return BrokerOrderRejected(
            order_id=order_id,
            broker_order_id=broker_order_id,
            reason="broker rejected (inactive)",
            occurred_at=occurred_at,
        )
    return None


def parse_execution_time(raw: str | None, *, fallback: datetime) -> datetime:
    """Parse IB's execution timestamp format, falling back to a supplied UTC time."""
    try:
        return datetime.strptime((raw or "").strip(), "%Y%m%d  %H:%M:%S").replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return fallback


def unmatched_fill_event_from_pending(
    *,
    pending: PendingExecution,
    con_id: int,
    occurred_at: datetime,
) -> BrokerUnmatchedFill:
    """Build the audit event for an IB fill without an internal instrument mapping."""
    return BrokerUnmatchedFill(
        ib_order_id=int(pending["ib_order_id"]),
        exec_id=str(pending["exec_id"]),
        con_id=con_id,
        occurred_at=occurred_at,
    )


def fill_event_from_pending(
    *,
    pending: PendingExecution,
    instrument_id: uuid.UUID,
    commission: Decimal,
    currency: str,
    received_at: datetime,
    fill_id: uuid.UUID,
) -> BrokerFillEvent:
    """Build a platform fill lifecycle event from pending execDetails + commissionReport."""
    ib_order_id = pending["ib_order_id"]
    internal_id = pending["internal_id"]
    if internal_id is None:
        raise ValueError("pending execution is missing an internal order id")
    executed_at = pending["time"] if isinstance(pending["time"], datetime) else received_at
    fill = FillEvent(
        fill_id=fill_id,
        order_id=internal_id,
        broker_order_id=str(ib_order_id),
        broker_execution_id=str(pending["exec_id"]),
        instrument_id=instrument_id,
        side=OrderSide.BUY if pending["side"] == "BOT" else OrderSide.SELL,
        quantity=int(pending["shares"]),
        fill_price=pending["price"],
        commission=commission,
        currency=currency,
        executed_at=executed_at,
        received_at=received_at,
    )
    return BrokerFillEvent(
        fill=fill,
        is_complete=False,
    )
