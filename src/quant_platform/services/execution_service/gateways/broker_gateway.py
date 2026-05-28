"""IB Gateway broker adapter implementing the BrokerGateway contract.

The facade owns IB client state, instrument mappings, and adapter composition.
Focused mixins provide connection lifecycle, account sync, order routing,
market-data, and lifecycle-feed behavior.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ibapi.client import EClient

from quant_platform.core.contracts import (
    BrokerAck,
    BrokerCapabilities,
    ExecutionPolicy,
)
from quant_platform.core.exceptions import BrokerUnavailableError
from quant_platform.services.execution_service.ib.ib_connection_lifecycle import (
    IBGatewayConnectionLifecycleMixin,
)
from quant_platform.services.execution_service.ib.ib_contract_mapper import (
    build_con_id_mapping,
    resolve_contract,
    validate_instrument_mappings,
)
from quant_platform.services.execution_service.ib.ib_gateway_account_sync import (
    IBGatewayAccountSyncMixin,
)
from quant_platform.services.execution_service.ib.ib_gateway_lifecycle_feed import (
    IBGatewayLifecycleFeedMixin,
)
from quant_platform.services.execution_service.ib.ib_gateway_market_data import (
    IBGatewayMarketDataMixin,
)
from quant_platform.services.execution_service.ib.ib_gateway_news import (
    IBGatewayNewsMixin,
)
from quant_platform.services.execution_service.ib.ib_gateway_order_routing import (
    IBGatewayOrderRoutingMixin,
)
from quant_platform.services.execution_service.ib.ib_historical_market_data import (
    IBHistoricalMarketDataRuntime,
)
from quant_platform.services.execution_service.ib.ib_news import IBNewsRuntime
from quant_platform.services.execution_service.ib_wrapper import _IBWrapper

if TYPE_CHECKING:
    import threading
    import uuid

    from ibapi.contract import Contract

    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.stores.pacing_store import HistoricalPacingStore

_TWS_LIVE_PORT = 7496
_TWS_PAPER_PORT = 7497
_GW_LIVE_PORT = 4001
_GW_PAPER_PORT = 4002


class IBGatewayBrokerGateway(
    IBGatewayConnectionLifecycleMixin,
    IBGatewayAccountSyncMixin,
    IBGatewayOrderRoutingMixin,
    IBGatewayMarketDataMixin,
    IBGatewayNewsMixin,
    IBGatewayLifecycleFeedMixin,
):
    """Live IB/TWS adapter with full ibapi integration."""

    def __init__(
        self,
        settings: BrokerSettings | None = None,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        paper_trading: bool = True,
        use_gateway: bool = False,
        client_id: int = 1,
        instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
        pacing_store: HistoricalPacingStore | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        if settings is not None:
            self._host = settings.host
            self._port = settings.port
            self._client_id = settings.client_id
            self._timeout = settings.request_timeout_seconds
        else:
            self._host = host
            if port is not None:
                self._port = port
            elif use_gateway:
                self._port = _GW_PAPER_PORT if paper_trading else _GW_LIVE_PORT
            else:
                self._port = _TWS_PAPER_PORT if paper_trading else _TWS_LIVE_PORT
            self._client_id = client_id
            self._timeout = 10.0

        self._connected = False
        self._submitted: dict[uuid.UUID, BrokerAck] = {}
        self._instrument_contracts = instrument_contracts or {}
        self._orphan_ttl_minutes = settings.orphan_ttl_minutes if settings is not None else 60
        self._execution_policy: ExecutionPolicy | None = execution_policy

        self._con_id_to_instrument = build_con_id_mapping(self._instrument_contracts)
        self._wrapper = _IBWrapper()
        self._client = EClient(self._wrapper)
        self._reader_thread: threading.Thread | None = None

        hist_enabled = False
        hist_pacing_window = 600.0
        hist_pacing_max = 60
        if settings is not None:
            hist_enabled = settings.historical_bar_fetch_enabled
            hist_pacing_window = settings.historical_bar_pacing_window_seconds
            hist_pacing_max = settings.historical_bar_pacing_max_requests
        self._historical_market_data = IBHistoricalMarketDataRuntime(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            client_id=self._client_id,
            enabled=hist_enabled,
            pacing_window_seconds=hist_pacing_window,
            pacing_max_requests=hist_pacing_max,
            pacing_store=pacing_store,
            instrument_contracts=self._instrument_contracts,
            require_connected=self._require_connected,
            resolve_contract=self._resolve_contract,
        )
        self._news_runtime = IBNewsRuntime(
            client=self._client,
            wrapper=self._wrapper,
            timeout=self._timeout,
            instrument_contracts=self._instrument_contracts,
            require_connected=self._require_connected,
        )

        self._order_id_lock = asyncio.Lock()
        self._max_local_order_id = 0
        self._capabilities = BrokerCapabilities(
            provider="tws",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=True,
        )

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self._capabilities

    def validate_instrument_mappings(self) -> list[str]:
        """Check that all mapped instruments have a con_id for reverse lookup."""
        return validate_instrument_mappings(self._instrument_contracts)

    def _resolve_instrument_id(self, con_id: int) -> uuid.UUID | None:
        """Resolve an IB conId to the canonical internal instrument_id."""
        return self._con_id_to_instrument.get(con_id)

    def set_execution_policy(self, policy: ExecutionPolicy) -> None:
        """Attach the ExecutionPolicy after session assembly."""
        self._execution_policy = policy

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerUnavailableError("IB Gateway is not connected; call connect() first")

    def _resolve_contract(self, instrument_id: uuid.UUID) -> Contract:
        """Resolve internal instrument_id to an IB Contract."""
        return resolve_contract(self._instrument_contracts, instrument_id)
