"""Order-state event helpers for account-level V2 execution."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
    VenueRoute,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from quant_platform.core.contracts import OrderStateStore


def route_event_payload(route: VenueRoute) -> dict[str, object]:
    """Serialize an EMS route to an order-state audit payload."""
    return {
        "route_id": str(route.route_id),
        "venue": route.venue,
        "tactic": route.tactic.value,
        "max_participation_rate": str(route.max_participation_rate),
        "urgency": str(route.urgency),
    }


async def append_order_state(
    order_state: OrderStateStore,
    intent: OrderIntent,
    event_type: OrderStateEventType,
    occurred_at: datetime,
    status: OrderStatus,
    *,
    payload: dict[str, object] | None = None,
) -> None:
    """Append one idempotent OMS lifecycle event."""
    await order_state.append(
        OrderStateEvent(
            event_id=uuid.uuid4(),
            order_id=intent.order_id,
            event_type=event_type,
            occurred_at=occurred_at,
            status=status,
            idempotency_key=f"{intent.order_id}:{event_type.value}",
            payload=payload or {},
        )
    )


async def append_created_events(
    order_state: OrderStateStore,
    intents: Sequence[OrderIntent],
    occurred_at: datetime,
) -> None:
    for intent in intents:
        await append_order_state(
            order_state,
            intent,
            OrderStateEventType.CREATED,
            occurred_at,
            OrderStatus.PENDING_APPROVAL,
        )


async def append_approval_events(
    order_state: OrderStateStore,
    approved: Sequence[OrderIntent],
    rejected: Sequence[OrderIntent],
    occurred_at: datetime,
) -> None:
    for intent in approved:
        await append_order_state(
            order_state,
            intent,
            OrderStateEventType.APPROVED,
            occurred_at,
            OrderStatus.APPROVED,
            payload={"cash_reservation_id": str(intent.cash_reservation_id)},
        )
    for intent in rejected:
        await append_order_state(
            order_state,
            intent,
            OrderStateEventType.REJECTED,
            occurred_at,
            OrderStatus.REJECTED,
        )


async def append_acknowledged_events(
    order_state: OrderStateStore,
    approved: Sequence[OrderIntent],
    submitted_ids: Sequence[uuid.UUID],
    occurred_at: datetime,
) -> None:
    submitted = set(submitted_ids)
    for intent in approved:
        if intent.order_id not in submitted:
            continue
        latest = await order_state.latest(intent.order_id)
        if latest is None or latest.event_type != OrderStateEventType.ACKNOWLEDGED:
            await append_order_state(
                order_state,
                intent,
                OrderStateEventType.ACKNOWLEDGED,
                occurred_at,
                OrderStatus.SUBMITTED,
            )


__all__ = [
    "append_acknowledged_events",
    "append_approval_events",
    "append_created_events",
    "append_order_state",
    "route_event_payload",
]
