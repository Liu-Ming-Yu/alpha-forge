"""Pure helper functions for cash-ledger accounting."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import FillEvent, OrderIntent, OrderSide
from quant_platform.core.domain.settlement import SettlementLot, SettlementStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import date

    from quant_platform.core.domain.portfolio.positions import PositionSnapshot

SETTLEMENT_LOT_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def price_for_cash_check(
    intent: OrderIntent,
    positions: Iterable[PositionSnapshot],
) -> Decimal | None:
    if intent.limit_price is not None:
        return intent.limit_price
    pos = next((item for item in positions if item.instrument_id == intent.instrument_id), None)
    return pos.market_price if pos else None


def required_cash_for_buy(quantity: int, price: Decimal, buffer_pct: Decimal) -> Decimal:
    return Decimal(str(quantity)) * price * (Decimal("1") + buffer_pct)


def settlement_lots_for_sell_fills(
    fills: Iterable[FillEvent],
    *,
    settlement_date_for: Callable[[date], date],
) -> list[SettlementLot]:
    lots: list[SettlementLot] = []
    for fill in fills:
        if fill.side != OrderSide.SELL:
            continue
        trade_date = fill.executed_at.date()
        gross = Decimal(str(fill.quantity)) * fill.fill_price
        net = gross - fill.commission
        lots.append(
            SettlementLot(
                lot_id=uuid.uuid5(SETTLEMENT_LOT_NAMESPACE, str(fill.fill_id)),
                fill_id=fill.fill_id,
                instrument_id=fill.instrument_id,
                trade_date=trade_date,
                settlement_date=settlement_date_for(trade_date),
                gross_proceeds=gross,
                commission=fill.commission,
                net_proceeds=net,
                currency=fill.currency,
                status=SettlementStatus.PENDING,
            )
        )
    return lots


def compact_uuid_set(values: set[uuid.UUID], threshold: int) -> int:
    """Drop the oldest half of a large UUID set and return removed count."""
    if len(values) <= threshold:
        return 0
    half = len(values) // 2
    for item in list(values)[:half]:
        values.discard(item)
    return half
