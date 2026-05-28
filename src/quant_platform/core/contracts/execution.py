"""Execution contracts: pacing/kill switch, broker gateway, lifecycle feed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts.common import (
        BrokerAck,
        BrokerCapabilities,
        BrokerHealth,
        TradeDecision,
    )
    from quant_platform.core.domain.orders import (
        BrokerOrder,
        CancelReplaceRequest,
        OrderIntent,
        OrderStateEvent,
        VenueRoute,
    )
    from quant_platform.core.domain.orders.lifecycle import BrokerLifecycleEvent
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot


@runtime_checkable
class ExecutionPolicy(Protocol):
    """Enforce pacing, kill switch, and submission rules for outbound orders.

    Must never:
        Submit orders directly; it only approves or blocks them.
        Be bypassed by any code path.

    Research-to-production parity requirement:
        The same implementation is used in both backtest and live runs.
        In backtest mode the kill switch and pacing are no-ops but the
        interface must be satisfied.
    """

    def can_submit(self, intent: OrderIntent) -> TradeDecision:
        """Check whether an approved OrderIntent may be submitted right now.

        Checks:
        - Kill switch is not active.
        - Throttle budget has not been exhausted.
        - Session is within market hours (if configured).

        Args:
            intent: The order seeking submission clearance.

        Returns:
            TradeDecision with approved=True if submission is permitted.
        """
        ...

    def record_submission(self, order_id: uuid.UUID) -> None:
        """Inform the policy that an order was submitted (for throttle accounting).

        Args:
            order_id: FK to the submitted OrderIntent.
        """
        ...

    @property
    def kill_switch_active(self) -> bool:
        """True if the kill switch has been activated."""
        ...


@runtime_checkable
class LifecycleFeed(Protocol):
    """Shared order-lifecycle event surface for fills, cancels, and rejects.

    Both IBGatewayBrokerGateway and SimulatedBrokerGateway implement this
    protocol so the AccountStateCoordinator can process broker events through
    a single, adapter-agnostic path.

    Must never:
        Block indefinitely; return immediately with whatever events have
        accumulated since the last call.
        Return the same event twice across two calls (events are consumed).

    Design rule:
        Every code path that processes broker order outcomes must go through
        drain_lifecycle_events() → AccountStateCoordinator.
    """

    async def drain_lifecycle_events(self) -> list[BrokerLifecycleEvent]:
        """Return and clear all pending lifecycle events.

        Returns accumulated events since the last call in arrival order.
        An empty list is returned when no events are pending.

        Events returned:
            BrokerFillEvent      — partial or complete fill
            BrokerOrderCancelled — order cancelled by the broker
            BrokerOrderRejected  — order rejected by the broker
            BrokerOrderCompleted — order fully executed (IB path only)
        """
        ...


@runtime_checkable
class BrokerGateway(Protocol):
    """Broker adapter boundary for IB Gateway-first execution.

    This is the only interface through which the execution service communicates
    with the broker.  All other code must treat the broker as unreachable.

    Must never:
        Be called outside the execution service.
        Return cached state as live data.
        Raise on transient broker errors without retrying via the retry policy.

    Idempotency:
        place_order() must accept repeated calls with the same order_id and
        return the existing BrokerAck rather than submitting a duplicate order.
    """

    async def connect(self) -> None:
        """Establish the broker session.  Idempotent if already connected."""
        ...

    async def disconnect(self) -> None:
        """Gracefully close the broker session."""
        ...

    async def health_check(self) -> BrokerHealth:
        """Return a current health report for the broker connection."""
        ...

    async def sync_account(self) -> AccountSnapshot:
        """Fetch the current account state directly from the broker.

        Returns a fresh AccountSnapshot.  Must never use cached data.
        """
        ...

    async def sync_positions(self) -> list[PositionSnapshot]:
        """Fetch current open positions directly from the broker."""
        ...

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        """Submit an order to the broker.

        Idempotent: repeated calls with the same order.order_id must not
        create duplicate broker orders.

        Args:
            order: The fully approved and reserved OrderIntent to submit.

        Returns:
            BrokerAck with the broker-assigned order ID.

        Failure semantics:
            Raises BrokerSubmissionError on non-retryable broker rejection.
            Raises BrokerUnavailableError on connection failure (caller retries).
        """
        ...

    async def cancel_order(self, broker_order_id: str) -> None:
        """Request cancellation of an open broker order.

        Args:
            broker_order_id: The broker's order identifier.

        Failure semantics:
            Raises BrokerOrderNotFoundError if the order does not exist.
            Raises BrokerUnavailableError on connection failure.
        """
        ...

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        """Return all open orders reported by the broker."""
        ...

    @property
    def capabilities(self) -> BrokerCapabilities:
        """Static adapter capability metadata for routing and safety checks."""
        ...


@runtime_checkable
class BrokerSessionGateway(Protocol):
    """Account/health/session-facing broker interface.

    Intended for routing account snapshot and connectivity concerns separately
    from order-routing concerns.
    """

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def health_check(self) -> BrokerHealth: ...
    async def sync_account(self) -> AccountSnapshot: ...
    async def sync_positions(self) -> list[PositionSnapshot]: ...
    async def fetch_open_orders(self) -> list[BrokerOrder]: ...

    @property
    def capabilities(self) -> BrokerCapabilities: ...


@runtime_checkable
class BrokerOrderRoutingGateway(Protocol):
    """Order-routing broker interface."""

    async def place_order(self, order: OrderIntent) -> BrokerAck: ...
    async def cancel_order(self, broker_order_id: str) -> None: ...

    @property
    def capabilities(self) -> BrokerCapabilities: ...


@runtime_checkable
class OrderStateStore(Protocol):
    """Event-sourced OMS state store."""

    async def append(self, event: OrderStateEvent) -> None:
        """Append one idempotent order-state event."""
        ...

    async def list_events(self, order_id: uuid.UUID) -> list[OrderStateEvent]:
        """Return all state events for an order in occurrence order."""
        ...

    async def latest(self, order_id: uuid.UUID) -> OrderStateEvent | None:
        """Return the latest state event for an order."""
        ...


@runtime_checkable
class ExecutionRouter(Protocol):
    """EMS router that maps approved intents to venue/tactic instructions."""

    def route(self, intent: OrderIntent) -> VenueRoute:
        """Return the route/tactic to use for this order."""
        ...

    async def cancel_replace(self, request: CancelReplaceRequest) -> None:
        """Submit a cancel/replace request through the execution adapter."""
        ...
