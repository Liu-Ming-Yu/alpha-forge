"""Account, position, and health sync behavior for the IB gateway."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.contracts import (
    BrokerHealth,
    BrokerHealthStatus,
)
from quant_platform.services.execution_service.ib.ib_account_sync import (
    sync_account_snapshot,
    sync_position_snapshots,
)

if TYPE_CHECKING:
    import uuid

    from ibapi.client import EClient

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
    from quant_platform.services.execution_service.ib_wrapper import _IBWrapper


class IBGatewayAccountSyncMixin:
    """Broker account/position sync methods for the IB gateway facade."""

    _client: EClient
    _connected: bool
    _timeout: float
    _wrapper: _IBWrapper

    def _require_connected(self) -> None:
        raise NotImplementedError

    def _resolve_instrument_id(self, con_id: int) -> uuid.UUID | None:
        raise NotImplementedError

    async def health_check(self) -> BrokerHealth:
        if not self._connected:
            return BrokerHealth(
                status=BrokerHealthStatus.DISCONNECTED,
                latency_ms=0,
                last_heartbeat_at=datetime.now(tz=UTC),
                detail="session not connected",
            )

        loop = asyncio.get_running_loop()
        self._wrapper._time_future = loop.create_future()
        start = loop.time()
        self._client.reqCurrentTime()
        try:
            await asyncio.wait_for(self._wrapper._time_future, timeout=self._timeout)
            latency = int((loop.time() - start) * 1000)
            return BrokerHealth(
                status=BrokerHealthStatus.CONNECTED,
                latency_ms=max(0, latency),
                last_heartbeat_at=datetime.now(tz=UTC),
            )
        except TimeoutError:
            return BrokerHealth(
                status=BrokerHealthStatus.DEGRADED,
                latency_ms=int(self._timeout * 1000),
                last_heartbeat_at=datetime.now(tz=UTC),
                detail="reqCurrentTime timed out",
            )

    async def sync_account(self) -> AccountSnapshot:
        self._require_connected()
        return await sync_account_snapshot(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            sync_positions=self.sync_positions,
        )

    async def sync_positions(self) -> list[PositionSnapshot]:
        self._require_connected()
        return await sync_position_snapshots(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            resolve_instrument_id=self._resolve_instrument_id,
        )
