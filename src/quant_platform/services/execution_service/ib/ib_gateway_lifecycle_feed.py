"""Lifecycle-feed delegates for the IB gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.execution_service.ib.ib_open_orders_sync import (
    fetch_open_orders_sync,
)

if TYPE_CHECKING:
    import uuid

    from ibapi.client import EClient

    from quant_platform.core.contracts import BrokerAck, ExecutionPolicy
    from quant_platform.core.domain.orders import BrokerOrder
    from quant_platform.core.domain.orders.lifecycle import BrokerLifecycleEvent
    from quant_platform.services.execution_service.ib_wrapper import _IBWrapper


class IBGatewayLifecycleFeedMixin:
    """Lifecycle event and open-order sync methods for the IB gateway facade."""

    _client: EClient
    _con_id_to_instrument: dict[int, uuid.UUID]
    _connected: bool
    _execution_policy: ExecutionPolicy | None
    _orphan_ttl_minutes: int
    _submitted: dict[uuid.UUID, BrokerAck]
    _timeout: float
    _wrapper: _IBWrapper

    async def drain_lifecycle_events(self) -> list[BrokerLifecycleEvent]:
        with self._wrapper._lifecycle_lock:
            events = list(self._wrapper._lifecycle_queue)
            self._wrapper._lifecycle_queue.clear()
        return events

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        if not self._connected:
            return []
        return await fetch_open_orders_sync(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            con_id_to_instrument=self._con_id_to_instrument,
            submitted=self._submitted,
            execution_policy=self._execution_policy,
            orphan_ttl_minutes=self._orphan_ttl_minutes,
        )
