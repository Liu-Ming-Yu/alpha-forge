"""Passive limit reprice DTOs and local protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.contracts import BrokerAck
    from quant_platform.core.domain.orders import BrokerOrder, OrderIntent, VenueRoute

PassiveRepriceAction = Literal["skipped", "cancelled", "replaced", "escalated", "failed"]


class PassiveRepriceBroker(Protocol):
    """Broker surface needed by passive reprice coordination."""

    async def fetch_open_orders(self) -> list[BrokerOrder]: ...
    async def place_order(self, order: OrderIntent) -> BrokerAck: ...


class PassiveReplacementFactory(Protocol):
    """Build a replacement intent after the original passive order is cancelled."""

    def __call__(
        self,
        original: OrderIntent,
        *,
        new_limit_price: Decimal,
        route: VenueRoute,
        requested_at: datetime,
    ) -> OrderIntent | None: ...


@dataclass(frozen=True)
class PassiveRepriceDecision:
    """Auditable result of evaluating one open order for passive repricing."""

    order_id: uuid.UUID
    action: PassiveRepriceAction
    reason: str
    broker_order_id: str | None = None
    replacement_order_id: uuid.UUID | None = None
    new_limit_price: Decimal | None = None
