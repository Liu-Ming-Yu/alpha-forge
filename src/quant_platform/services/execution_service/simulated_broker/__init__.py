"""Simulated broker gateway for backtest, paper trading, and unit testing.

SimulatedBrokerGateway satisfies the BrokerGateway and LifecycleFeed protocols
without any external broker dependency.  It models deterministic order fills at
the limit price (or a configurable market price), with optional fill-price and
commission adjustment callbacks, and maintains internal account and position state.

Lifecycle event contract:
    Every place_order() call enqueues a BrokerFillEvent on the internal
    lifecycle queue.  drain_lifecycle_events() drains and returns the queue.

Partial fill simulation (for testing):
    simulate_partial_fill() lets tests inject a partial fill for a submitted
    order without fully completing it.  Call again with is_complete=True to
    close the position.

Cancel/reject simulation:
    simulate_cancel() and simulate_reject() let tests inject broker-initiated
    cancel and reject events.

Research-to-production parity:
    This adapter is the ONLY difference between backtest and live.  All
    other components (CashLedger, RiskPolicy, ExecutionPolicy, controllers)
    are identical.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.contracts import (
    BrokerAck,
    BrokerCapabilities,
    BrokerHealth,
    BrokerHealthStatus,
    Clock,
)
from quant_platform.services.execution_service.simulated_broker.simulated_broker_account import (
    SimulatedBrokerAccountMixin,
)
from quant_platform.services.execution_service.simulated_broker.simulated_broker_controls import (
    SimulatedBrokerControlsMixin,
)
from quant_platform.services.execution_service.simulated_broker.simulated_broker_orders import (
    SimulatedBrokerOrdersMixin,
)
from quant_platform.services.execution_service.simulated_broker.simulated_fill_model import (
    ParticipationFillModel,
    SimulatedFillPlan,
    SimulatedLiquidityProfile,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.core.domain.orders import (
        OrderIntent,
    )
    from quant_platform.core.domain.orders.lifecycle import (
        BrokerLifecycleEvent,
    )

log = structlog.get_logger(__name__)

__all__ = [
    "ParticipationFillModel",
    "SimulatedBrokerGateway",
    "SimulatedFillPlan",
    "SimulatedLiquidityProfile",
]


class SimulatedBrokerGateway(
    SimulatedBrokerOrdersMixin,
    SimulatedBrokerControlsMixin,
    SimulatedBrokerAccountMixin,
):
    """Deterministic simulated broker for backtest and testing.

    Args:
        clock: Injectable time source.
        initial_cash: Starting cash balance.
        fill_at_limit: If True, orders fill at their limit price.
            If False, fills at the last known market price for the instrument.

    Simulated fill behaviour:
        - Limit orders fill at the limit price (or adjusted price when
          configure_execution_cost_model() is configured).
        - Market orders fill at the last known price for the instrument.
        - By default all fills are complete (no partial fills).
        - Use simulate_partial_fill() to inject partial-fill scenarios.
    """

    def __init__(
        self,
        clock: Clock,
        initial_cash: Decimal = Decimal("100000"),
        fill_at_limit: bool = True,
    ) -> None:
        self._clock = clock
        self._fill_at_limit = fill_at_limit
        self._connected = False
        self._settled_cash = initial_cash
        self._positions: dict[uuid.UUID, int] = {}
        self._avg_costs: dict[uuid.UUID, Decimal] = {}
        self._submitted: dict[uuid.UUID, BrokerAck] = {}
        self._open_orders: dict[uuid.UUID, OrderIntent] = {}
        self._lifecycle_queue: list[BrokerLifecycleEvent] = []
        # Prefix with a per-instance random token so that concurrent test
        # runs against a shared Postgres DB don't collide on the partial
        # unique index (broker_order_id, broker_execution_id).
        self._broker_id_prefix = uuid.uuid4().hex[:8]
        self._next_ib_id = 1000
        self._next_execution_id = 1
        self._market_prices: dict[uuid.UUID, Decimal] = {}
        self._fill_price_adjuster: Callable[[OrderIntent, Decimal], Decimal] | None = None
        self._commission_calculator: Callable[[OrderIntent, Decimal], Decimal] | None = None
        self._execution_model: ParticipationFillModel | None = None
        self._execution_plans: dict[uuid.UUID, SimulatedFillPlan] = {}

        self._capabilities = BrokerCapabilities(
            provider="simulated",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=True,
        )

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self._capabilities

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_market_price(self, instrument_id: uuid.UUID, price: Decimal) -> None:
        """Set the simulated market price for an instrument."""
        self._market_prices[instrument_id] = price

    def configure_execution_cost_model(
        self,
        *,
        fill_price_adjuster: Callable[[OrderIntent, Decimal], Decimal] | None = None,
        commission_calculator: Callable[[OrderIntent, Decimal], Decimal] | None = None,
    ) -> None:
        """Configure optional fill-price and commission adjustment callbacks."""
        self._fill_price_adjuster = fill_price_adjuster
        self._commission_calculator = commission_calculator

    def configure_execution_model(self, model: ParticipationFillModel | None) -> None:
        """Configure a richer simulated execution model.

        When set, ``place_order`` uses the model to determine fill
        quantity, price, commission, and completion status.  When unset,
        the gateway preserves its historical immediate full-fill behavior.
        """
        self._execution_model = model

    def execution_plan_for(self, order_id: uuid.UUID) -> SimulatedFillPlan | None:
        """Return the most recent simulated execution plan for ``order_id``."""
        return self._execution_plans.get(order_id)

    def _next_broker_execution_id(self, broker_order_id: str) -> str:
        execution_id = f"sim-{broker_order_id}-{self._next_execution_id}"
        self._next_execution_id += 1
        return execution_id

    # ------------------------------------------------------------------
    # BrokerGateway protocol
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def health_check(self) -> BrokerHealth:
        return BrokerHealth(
            status=BrokerHealthStatus.CONNECTED
            if self._connected
            else BrokerHealthStatus.DISCONNECTED,
            latency_ms=0,
            last_heartbeat_at=self._clock.now(),
        )

    # ------------------------------------------------------------------
    # LifecycleFeed protocol
    # ------------------------------------------------------------------

    async def drain_lifecycle_events(self) -> list[BrokerLifecycleEvent]:
        """Return and clear all pending lifecycle events.

        This is the single event-delivery path used by simulated and live
        broker adapters.
        """
        events = list(self._lifecycle_queue)
        self._lifecycle_queue.clear()
        return events
