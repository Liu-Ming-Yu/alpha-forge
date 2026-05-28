"""Simulation-control helpers for the in-process broker adapter."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import FillEvent, OrderIntent, OrderSide
from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerOrderCancelled,
    BrokerOrderRejected,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import BrokerAck, Clock
    from quant_platform.core.domain.orders.lifecycle import BrokerLifecycleEvent


class SimulatedBrokerControlsMixin:
    """Test controls and internal fill accounting for simulated broker."""

    _clock: Clock
    _settled_cash: Decimal
    _positions: dict[uuid.UUID, int]
    _avg_costs: dict[uuid.UUID, Decimal]
    _submitted: dict[uuid.UUID, BrokerAck]
    _open_orders: dict[uuid.UUID, OrderIntent]
    _lifecycle_queue: list[BrokerLifecycleEvent]

    if TYPE_CHECKING:

        def _next_broker_execution_id(self, broker_order_id: str) -> str: ...

    def simulate_partial_fill(
        self,
        order_id: uuid.UUID,
        quantity: int,
        price: Decimal,
        is_complete: bool = False,
        commission: Decimal = Decimal("0.50"),
    ) -> FillEvent:
        """Inject a partial or final fill for a submitted order."""
        ack = self._submitted[order_id]
        order = self._open_orders.get(order_id)
        now = self._clock.now()

        fill = FillEvent(
            fill_id=uuid.uuid4(),
            order_id=order_id,
            broker_order_id=ack.broker_order_id,
            broker_execution_id=self._next_broker_execution_id(ack.broker_order_id),
            instrument_id=order.instrument_id if order else uuid.uuid4(),
            side=order.side if order else OrderSide.BUY,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            currency="USD",
            executed_at=now,
            received_at=now,
        )

        self._apply_fill_to_internal(fill)

        if is_complete and order_id in self._open_orders:
            self._open_orders.pop(order_id, None)

        self._lifecycle_queue.append(BrokerFillEvent(fill=fill, is_complete=is_complete))
        return fill

    def simulate_cancel(
        self,
        order_id: uuid.UUID,
        reason: str = "simulated cancel",
    ) -> None:
        """Inject a broker-initiated cancel for a submitted order."""
        ack = self._submitted[order_id]
        self._open_orders.pop(order_id, None)
        self._lifecycle_queue.append(
            BrokerOrderCancelled(
                order_id=order_id,
                broker_order_id=ack.broker_order_id,
                reason=reason,
                occurred_at=self._clock.now(),
            )
        )

    def simulate_reject(
        self,
        order_id: uuid.UUID,
        reason: str = "simulated reject",
    ) -> None:
        """Inject a broker-initiated rejection for a submitted order."""
        ack = self._submitted[order_id]
        self._open_orders.pop(order_id, None)
        self._lifecycle_queue.append(
            BrokerOrderRejected(
                order_id=order_id,
                broker_order_id=ack.broker_order_id,
                reason=reason,
                occurred_at=self._clock.now(),
            )
        )

    def _apply_fill_to_internal(self, fill: FillEvent) -> None:
        """Update the broker's internal cash and position state for a fill."""
        if fill.side == OrderSide.BUY:
            cost = Decimal(str(fill.quantity)) * fill.fill_price + fill.commission
            self._settled_cash -= cost
            prev_qty = self._positions.get(fill.instrument_id, 0)
            prev_cost = self._avg_costs.get(fill.instrument_id, Decimal("0"))
            new_qty = prev_qty + fill.quantity
            if new_qty > 0:
                self._avg_costs[fill.instrument_id] = (
                    prev_cost * prev_qty + fill.fill_price * fill.quantity
                ) / new_qty
            self._positions[fill.instrument_id] = new_qty
        else:
            proceeds = Decimal(str(fill.quantity)) * fill.fill_price - fill.commission
            self._settled_cash += proceeds
            self._positions[fill.instrument_id] = (
                self._positions.get(fill.instrument_id, 0) - fill.quantity
            )
            if self._positions.get(fill.instrument_id, 0) <= 0:
                self._positions.pop(fill.instrument_id, None)
                self._avg_costs.pop(fill.instrument_id, None)
