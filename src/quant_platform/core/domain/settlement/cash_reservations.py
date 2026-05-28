"""Cash reservation domain model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal


class ReservationStatus(StrEnum):
    ACTIVE = "active"
    RELEASED = "released"  # Released because order was filled
    EXPIRED = "expired"  # Released because order was cancelled/rejected/expired
    FAILED = "failed"  # Released because reservation creation itself failed


@dataclass(frozen=True)
class CashReservation:
    """A hold on settled cash earmarked for a pending buy order.

    Created by CashConstraintEngine.reserve_cash() before an OrderIntent is
    promoted to a BrokerOrder.  Released by release_reservation() when the
    order lifecycle terminates (fill, cancel, reject, or expiry).

    Args:
        reservation_id: Stable system UUID.
        order_id: FK to OrderIntent.order_id.  Unique among ACTIVE reservations.
        reserved_amount: USD amount reserved (quantity × estimated_price + buffer).
        reserved_at: UTC timestamp of reservation creation.
        expires_at: UTC timestamp after which this reservation auto-expires if
            the order has not been submitted.  Prevents leaked reservations.
        status: Reservation lifecycle status.
        released_at: UTC timestamp of release.  None until released/expired.
        release_reason: Human-readable reason for release (e.g. "order filled",
            "order cancelled by operator").

    Invariants:
        reserved_amount > 0.
        Two ACTIVE reservations must not share the same order_id.
        released_at must be set when status transitions to RELEASED or EXPIRED.
    """

    reservation_id: uuid.UUID
    order_id: uuid.UUID
    reserved_amount: Decimal
    reserved_at: datetime
    expires_at: datetime
    status: ReservationStatus
    released_at: datetime | None = None
    release_reason: str = ""

    def __post_init__(self) -> None:
        if self.reserved_amount <= 0:
            raise ValueError("reserved_amount must be > 0")
        if self.expires_at <= self.reserved_at:
            raise ValueError("expires_at must be after reserved_at")
        if self.reserved_at.tzinfo is None:
            raise ValueError("reserved_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if (
            self.status in (ReservationStatus.RELEASED, ReservationStatus.EXPIRED)
            and self.released_at is None
        ):
            raise ValueError("released_at must be set when status is RELEASED or EXPIRED")
