"""Account projection behavior for the simulated broker adapter."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

if TYPE_CHECKING:
    from quant_platform.core.contracts import Clock


class SimulatedBrokerAccountMixin:
    """Build account and position snapshots from simulated broker state."""

    _clock: Clock
    _settled_cash: Decimal
    _positions: dict[uuid.UUID, int]
    _avg_costs: dict[uuid.UUID, Decimal]
    _market_prices: dict[uuid.UUID, Decimal]

    async def sync_account(self) -> AccountSnapshot:
        now = self._clock.now()
        pos_snapshots = await self.sync_positions()
        market_value = sum(Decimal(str(p.quantity)) * p.market_price for p in pos_snapshots)
        return AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=now,
            settled_cash=self._settled_cash,
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=self._settled_cash,
            net_asset_value=self._settled_cash + market_value,
            positions=tuple(pos_snapshots),
            source="simulated",
        )

    async def sync_positions(self) -> list[PositionSnapshot]:
        now = self._clock.now()
        snapshots: list[PositionSnapshot] = []
        for inst_id, qty in self._positions.items():
            if qty <= 0:
                continue
            avg_cost = self._avg_costs.get(inst_id, Decimal("1"))
            market_price = self._market_prices.get(inst_id, avg_cost)
            market_value = Decimal(str(qty)) * market_price
            unrealised = market_value - Decimal(str(qty)) * avg_cost
            snapshots.append(
                PositionSnapshot(
                    snapshot_id=uuid.uuid4(),
                    instrument_id=inst_id,
                    quantity=qty,
                    average_cost=avg_cost,
                    market_price=market_price,
                    market_value=market_value,
                    unrealised_pnl=unrealised,
                    as_of=now,
                    source="simulated",
                )
            )
        return snapshots


__all__ = ["SimulatedBrokerAccountMixin"]
