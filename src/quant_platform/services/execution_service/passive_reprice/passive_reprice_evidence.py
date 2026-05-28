"""OMS evidence helpers for passive reprice decisions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import (
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import Clock, OrderStateStore
    from quant_platform.services.execution_service.passive_reprice.passive_reprice_models import (
        PassiveRepriceDecision,
    )


async def append_passive_reprice_evidence(
    *,
    order_state: OrderStateStore | None,
    clock: Clock,
    decision: PassiveRepriceDecision,
) -> None:
    """Append one OMS evidence row for a passive-reprice decision."""
    if order_state is None:
        return
    event_type = OrderStateEventType.ROUTED
    if decision.action in {"cancelled", "replaced", "escalated", "failed"}:
        event_type = OrderStateEventType.CANCEL_REQUESTED
    now = clock.now()
    await order_state.append(
        OrderStateEvent(
            event_id=uuid.uuid4(),
            order_id=decision.order_id,
            event_type=event_type,
            occurred_at=now,
            status=OrderStatus.SUBMITTED,
            broker_order_id=decision.broker_order_id,
            idempotency_key=(
                f"{decision.order_id}:passive_reprice:{decision.action}:{now.isoformat()}"
            ),
            payload={
                "source": "passive_reprice",
                "action": decision.action,
                "reason": decision.reason,
                "replacement_order_id": str(decision.replacement_order_id)
                if decision.replacement_order_id
                else "",
                "new_limit_price": str(decision.new_limit_price)
                if decision.new_limit_price is not None
                else "",
            },
        )
    )
