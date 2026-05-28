"""Value types shared across contract signatures.

These are small frozen dataclasses / enums used as arguments or return
values in multiple bounded-context contracts.  Keeping them in a single
file avoids circular imports when a contract module needs to reference
another context's helper type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal


class BrokerHealthStatus(StrEnum):
    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class BrokerHealth:
    """Point-in-time health report from the broker gateway.

    Args:
        status: Current connection health.
        latency_ms: Round-trip latency to broker, in milliseconds.  Stored as
            an integer so observability/round-trip serialisation is exact;
            sub-millisecond resolution is below the IBKR API's own jitter.
        last_heartbeat_at: UTC timestamp of last successful heartbeat.
        detail: Optional human-readable detail from the broker session layer.
    """

    status: BrokerHealthStatus
    latency_ms: int
    last_heartbeat_at: datetime
    detail: str = ""

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")


@dataclass(frozen=True)
class BrokerAck:
    """Acknowledgement returned by BrokerGateway.place_order().

    Args:
        order_id: Internal order UUID (echoed for correlation).
        broker_order_id: Broker-assigned order identifier.
        acknowledged_at: UTC timestamp of broker acknowledgement.
    """

    order_id: uuid.UUID
    broker_order_id: str
    acknowledged_at: datetime


@dataclass(frozen=True)
class BrokerCapabilities:
    """Capabilities advertised by a broker adapter.

    This metadata lets orchestration safely route account/health calls and
    order-routing calls without assuming every adapter supports both.
    """

    provider: str
    supports_order_routing: bool
    supports_order_cancellation: bool
    supports_lifecycle_feed: bool


@dataclass(frozen=True)
class TradeDecision:
    """Result of a CashConstraintEngine eligibility check.

    Args:
        approved: True if the order may proceed to reservation.
        reason: Human-readable explanation (required when approved=False).
        available_cash: Settled cash available at the time of the check.
        required_cash: Estimated cash required for the order.
    """

    approved: bool
    reason: str
    available_cash: Decimal
    required_cash: Decimal
