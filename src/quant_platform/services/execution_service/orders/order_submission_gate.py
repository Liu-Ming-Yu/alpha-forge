"""Pure submission gates for ``OrderThrottle``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import OrderIntent, OrderType

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.services.execution_service.support.trading_calendar import TradingCalendar

MOC_LOC_TYPES = frozenset({OrderType.MOC, OrderType.LOC})


def duplicate_order_reason(order_id: uuid.UUID, submitted_ids: set[uuid.UUID]) -> str | None:
    return "duplicate_order_id" if order_id in submitted_ids else None


def market_hours_reason(
    intent: OrderIntent,
    *,
    now: datetime,
    enforced: bool,
    calendar: TradingCalendar | None,
) -> str | None:
    if not enforced or calendar is None or intent.order_type in MOC_LOC_TYPES:
        return None
    return None if calendar.is_open(now) else "market_closed"
