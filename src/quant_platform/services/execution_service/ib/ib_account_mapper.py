"""IB account and position mapping helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime


def account_snapshot_from_values(
    *,
    snapshot_id: uuid.UUID,
    as_of: datetime,
    values: Mapping[str, str],
    positions: Sequence[PositionSnapshot],
) -> AccountSnapshot:
    """Translate IB account-summary values into a domain account snapshot."""
    settled = Decimal(values.get("SettledCash", values.get("TotalCashValue", "0")))
    nav = Decimal(values.get("NetLiquidation", "0"))
    return AccountSnapshot(
        snapshot_id=snapshot_id,
        as_of=as_of,
        settled_cash=settled,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=nav,
        positions=tuple(positions),
        source="broker",
    )


def position_snapshot_from_values(
    *,
    snapshot_id: uuid.UUID,
    instrument_id: uuid.UUID,
    quantity: int,
    average_cost: Decimal,
    as_of: datetime,
) -> PositionSnapshot:
    """Translate one broker position row into a domain position snapshot."""
    market_value = Decimal(str(quantity)) * average_cost
    return PositionSnapshot(
        snapshot_id=snapshot_id,
        instrument_id=instrument_id,
        quantity=quantity,
        average_cost=average_cost,
        market_price=average_cost,
        market_value=market_value,
        unrealised_pnl=Decimal("0"),
        as_of=as_of,
        source="broker",
    )
