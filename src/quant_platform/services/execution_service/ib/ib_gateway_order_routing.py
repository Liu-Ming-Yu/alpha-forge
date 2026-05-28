"""Order routing behavior for the IB gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.core.exceptions import BrokerUnavailableError
from quant_platform.services.execution_service.ib.ib_order_submission import (
    cancel_order_and_wait,
    submit_order_and_wait_for_ack,
)

if TYPE_CHECKING:
    import asyncio
    import uuid

    from ibapi.client import EClient
    from ibapi.contract import Contract

    from quant_platform.core.contracts import BrokerAck
    from quant_platform.core.domain.orders import OrderIntent

log = structlog.get_logger(__name__)


class IBGatewayOrderRoutingMixin:
    """Order submission and cancellation methods for the IB gateway facade."""

    _client: EClient
    _execution_policy: Any
    _max_local_order_id: int
    _order_id_lock: asyncio.Lock
    _submitted: dict[uuid.UUID, BrokerAck]
    _timeout: float
    _wrapper: Any

    def _require_connected(self) -> None:
        raise NotImplementedError

    def _resolve_contract(self, instrument_id: uuid.UUID) -> Contract:
        raise NotImplementedError

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        self._require_connected()

        if order.order_id in self._submitted:
            log.info("broker_gateway.place_order.idempotent", order_id=str(order.order_id))
            return self._submitted[order.order_id]

        if self._wrapper._next_order_id is None:
            raise BrokerUnavailableError("next valid order ID not yet received from IB")

        async with self._order_id_lock:
            ib_order_id = self._wrapper._next_order_id
            self._wrapper._next_order_id += 1
            self._max_local_order_id = max(self._max_local_order_id, ib_order_id)

        return await submit_order_and_wait_for_ack(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            order=order,
            ib_order_id=ib_order_id,
            contract=self._resolve_contract(order.instrument_id),
            submitted=self._submitted,
            execution_policy=self._execution_policy,
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        self._require_connected()
        await cancel_order_and_wait(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            broker_order_id=broker_order_id,
        )
