"""Client Portal broker stub adapter.

This is a non-routing adapter used to separate session/account concerns from
order-routing concerns while CP trading support is not yet implemented.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import (
    BrokerAck,
    BrokerCapabilities,
    BrokerHealth,
    BrokerHealthStatus,
    Clock,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.exceptions import BrokerSubmissionError

if TYPE_CHECKING:
    from quant_platform.core.contracts import BrokerSessionGateway
    from quant_platform.core.domain.orders import BrokerOrder, OrderIntent


class ClientPortalBrokerGateway:
    """Session/account-capable broker adapter with no order-routing support.

    Args:
        clock: Time source.
        initial_snapshot: Fallback account snapshot used when no upstream
            session gateway is configured.
        upstream_session_gateway: Optional upstream broker session adapter.
            When supplied, account/position/health calls proxy to upstream.
    """

    def __init__(
        self,
        clock: Clock,
        initial_snapshot: AccountSnapshot,
        upstream_session_gateway: BrokerSessionGateway | None = None,
    ) -> None:
        self._clock = clock
        self._connected = False
        self._snapshot = initial_snapshot
        self._upstream = upstream_session_gateway
        self._capabilities = BrokerCapabilities(
            provider="client_portal_stub",
            supports_order_routing=False,
            supports_order_cancellation=False,
            supports_lifecycle_feed=False,
        )

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self._capabilities

    async def connect(self) -> None:
        self._connected = True
        if self._upstream is not None:
            await self._upstream.connect()

    async def disconnect(self) -> None:
        self._connected = False
        if self._upstream is not None:
            await self._upstream.disconnect()

    async def health_check(self) -> BrokerHealth:
        if self._upstream is not None:
            return await self._upstream.health_check()

        status = (
            BrokerHealthStatus.CONNECTED if self._connected else BrokerHealthStatus.DISCONNECTED
        )
        return BrokerHealth(
            status=status,
            latency_ms=5,
            last_heartbeat_at=self._clock.now(),
            detail="client portal stub",
        )

    async def sync_account(self) -> AccountSnapshot:
        if self._upstream is not None:
            return await self._upstream.sync_account()
        return AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=self._clock.now(),
            settled_cash=self._snapshot.settled_cash,
            unsettled_cash=self._snapshot.unsettled_cash,
            reserved_cash=Decimal("0"),
            available_cash=self._snapshot.settled_cash,
            net_asset_value=self._snapshot.net_asset_value,
            positions=self._snapshot.positions,
            source="client_portal_stub",
        )

    async def sync_positions(self) -> list[PositionSnapshot]:
        if self._upstream is not None:
            return await self._upstream.sync_positions()
        return list(self._snapshot.positions)

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        if self._upstream is not None:
            return await self._upstream.fetch_open_orders()
        return []

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        raise BrokerSubmissionError(
            "ClientPortalBrokerGateway stub does not support order routing; "
            "use TWS/IB Gateway route for place_order"
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        raise BrokerSubmissionError(
            "ClientPortalBrokerGateway stub does not support order cancellation; "
            "use TWS/IB Gateway route for cancel_order"
        )
