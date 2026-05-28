"""Order-flow behavior for the simulated broker adapter."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import BrokerAck, Clock
from quant_platform.core.domain.orders import BrokerOrder, FillEvent, OrderIntent, OrderStatus
from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerLifecycleEvent,
    BrokerOrderCancelled,
    BrokerOrderRejected,
)
from quant_platform.core.exceptions import BrokerOrderNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.services.execution_service.simulated_broker.simulated_fill_model import (
        ParticipationFillModel,
        SimulatedFillPlan,
    )


class SimulatedBrokerOrdersMixin:
    """Order submission, cancellation, and open-order projection."""

    _clock: Clock
    _broker_id_prefix: str
    _next_ib_id: int
    _submitted: dict[uuid.UUID, BrokerAck]
    _open_orders: dict[uuid.UUID, OrderIntent]
    _market_prices: dict[uuid.UUID, Decimal]
    _execution_model: ParticipationFillModel | None
    _execution_plans: dict[uuid.UUID, SimulatedFillPlan]
    _fill_price_adjuster: Callable[[OrderIntent, Decimal], Decimal] | None
    _commission_calculator: Callable[[OrderIntent, Decimal], Decimal] | None
    _lifecycle_queue: list[BrokerLifecycleEvent]

    if TYPE_CHECKING:

        def _next_broker_execution_id(self, broker_order_id: str) -> str: ...

        def _apply_fill_to_internal(self, fill: FillEvent) -> None: ...

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        """Submit an order; immediately fills it at the limit or market price."""
        if order.order_id in self._submitted:
            return self._submitted[order.order_id]

        broker_order_id = f"{self._broker_id_prefix}-{self._next_ib_id}"
        self._next_ib_id += 1

        ack = BrokerAck(
            order_id=order.order_id,
            broker_order_id=broker_order_id,
            acknowledged_at=self._clock.now(),
        )
        self._submitted[order.order_id] = ack
        self._open_orders[order.order_id] = order

        reference_price = order.limit_price or self._market_prices.get(
            order.instrument_id, Decimal("100")
        )
        execution_model = self._execution_model
        if execution_model is not None:
            return self._place_with_execution_model(order, ack, reference_price)

        return self._place_immediate_fill(order, ack, reference_price)

    def _place_with_execution_model(
        self,
        order: OrderIntent,
        ack: BrokerAck,
        reference_price: Decimal,
    ) -> BrokerAck:
        execution_model = self._execution_model
        if execution_model is None:
            raise RuntimeError("simulated execution model is not configured")
        plan = execution_model.plan(order, reference_price)
        self._execution_plans[order.order_id] = plan

        if plan.filled_quantity <= 0:
            self._open_orders.pop(order.order_id, None)
            self._lifecycle_queue.append(
                BrokerOrderRejected(
                    order_id=order.order_id,
                    broker_order_id=ack.broker_order_id,
                    reason="simulated execution model filled 0 shares",
                    occurred_at=self._clock.now(),
                )
            )
            return ack

        fill = self._build_fill(
            order=order,
            broker_order_id=ack.broker_order_id,
            quantity=plan.filled_quantity,
            fill_price=plan.fill_price,
            commission=plan.commission,
        )
        self._record_fill(fill, is_complete=plan.is_complete)
        if plan.is_complete:
            self._open_orders.pop(order.order_id, None)
        return ack

    def _place_immediate_fill(
        self,
        order: OrderIntent,
        ack: BrokerAck,
        reference_price: Decimal,
    ) -> BrokerAck:
        fill_price = reference_price
        if self._fill_price_adjuster is not None:
            fill_price = self._fill_price_adjuster(order, reference_price)
        commission = Decimal("1.00")
        if self._commission_calculator is not None:
            commission = self._commission_calculator(order, fill_price)

        fill = self._build_fill(
            order=order,
            broker_order_id=ack.broker_order_id,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
        )
        self._record_fill(fill, is_complete=True)
        self._open_orders.pop(order.order_id, None)
        return ack

    def _build_fill(
        self,
        *,
        order: OrderIntent,
        broker_order_id: str,
        quantity: int,
        fill_price: Decimal,
        commission: Decimal,
    ) -> FillEvent:
        return FillEvent(
            fill_id=uuid.uuid4(),
            order_id=order.order_id,
            broker_order_id=broker_order_id,
            broker_execution_id=self._next_broker_execution_id(broker_order_id),
            instrument_id=order.instrument_id,
            side=order.side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            currency="USD",
            executed_at=self._clock.now(),
            received_at=self._clock.now(),
        )

    def _record_fill(self, fill: FillEvent, *, is_complete: bool) -> None:
        self._apply_fill_to_internal(fill)
        self._lifecycle_queue.append(BrokerFillEvent(fill=fill, is_complete=is_complete))

    async def cancel_order(self, broker_order_id: str) -> None:
        """Cancel an open order by broker order ID."""
        target = None
        for oid, ack in self._submitted.items():
            if ack.broker_order_id == broker_order_id and oid in self._open_orders:
                target = oid
                break
        if target is None:
            raise BrokerOrderNotFoundError(f"No open order with broker_order_id={broker_order_id}")
        self._open_orders.pop(target, None)
        self._lifecycle_queue.append(
            BrokerOrderCancelled(
                order_id=target,
                broker_order_id=broker_order_id,
                reason="operator cancel",
                occurred_at=self._clock.now(),
            )
        )

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        now = self._clock.now()
        return [
            BrokerOrder(
                order_id=oid,
                status=OrderStatus.SUBMITTED,
                last_updated_at=now,
                broker_order_id=self._submitted[oid].broker_order_id,
            )
            for oid in self._open_orders
        ]
