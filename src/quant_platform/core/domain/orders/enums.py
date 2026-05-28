"""Order lifecycle enums."""

from __future__ import annotations

from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    MOC = "moc"  # Market-on-close
    LOC = "loc"  # Limit-on-close


class OrderStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStateEventType(StrEnum):
    """Event-sourced OMS lifecycle event type."""

    CREATED = "created"
    APPROVED = "approved"
    ROUTED = "routed"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RECONCILED = "reconciled"
    UNCERTAIN = "uncertain"


class ExecutionTactic(StrEnum):
    """EMS tactic selected by the account-level execution router."""

    PASSIVE_LIMIT = "passive_limit"
    URGENCY_LIMIT = "urgency_limit"
    CLOSE_AUCTION_MOC = "close_auction_moc"
    CLOSE_AUCTION_LOC = "close_auction_loc"
