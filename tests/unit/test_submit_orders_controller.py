"""Unit tests for SubmitOrdersControllerImpl."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.contracts import BrokerAck, BrokerCapabilities, TradeDecision
from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderStateEventType,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.events import (
    KillSwitchActivated,
    OrderRejected,
    OrderSubmissionUncertain,
)
from quant_platform.core.exceptions import (
    BrokerAckTimeoutError,
    BrokerSubmissionError,
    BrokerUnavailableError,
)
from quant_platform.infrastructure.event_bus import InMemoryEventBus
from quant_platform.infrastructure.repositories import InMemoryOrderRepository
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.v2.state import InMemoryOrderStateStore
from quant_platform.services.execution_service.orders.controllers import SubmitOrdersControllerImpl

_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=UTC)


class _NoRoutingBroker:
    capabilities = BrokerCapabilities(
        provider="cp_stub",
        supports_order_routing=False,
        supports_order_cancellation=False,
        supports_lifecycle_feed=False,
    )

    async def place_order(self, order: OrderIntent) -> object:
        raise RuntimeError("should not be called")

    async def cancel_order(self, broker_order_id: str) -> None:
        raise RuntimeError("should not be called")


class _RoutingBroker:
    """Broker stub that passes the ``supports_order_routing`` short-circuit
    so downstream gates (risk revalidation, reservation staleness) are
    exercised.  ``place_order`` is not expected to be called in these
    paths — the gate should reject first."""

    capabilities = BrokerCapabilities(
        provider="routing_stub",
        supports_order_routing=True,
        supports_order_cancellation=True,
        supports_lifecycle_feed=False,
    )

    async def place_order(self, order: OrderIntent) -> object:
        raise RuntimeError("place_order should not run when the pre-flight gate rejects")

    async def cancel_order(self, broker_order_id: str) -> None:
        raise RuntimeError("should not be called")


class _AlwaysApprovePolicy:
    @property
    def kill_switch_active(self) -> bool:
        return False

    def can_submit(self, intent: OrderIntent) -> TradeDecision:
        return TradeDecision(
            approved=True,
            reason="ok",
            available_cash=Decimal("0"),
            required_cash=Decimal("0"),
        )

    def record_submission(self, order_id: uuid.UUID) -> None:
        return None


class _DurableKillSwitchPolicy(_AlwaysApprovePolicy):
    def __init__(self) -> None:
        self.activations: list[tuple[str, str]] = []

    async def activate_kill_switch_durable(self, reason: str, *, activated_by: str) -> None:
        self.activations.append((reason, activated_by))


class _CashStub:
    def cancel_order(self, order_id: uuid.UUID, reason: str) -> None:
        return None


class _ReservationAwareCashStub:
    """Cash stub that tracks reservation state and release calls."""

    def __init__(self, *, is_active: bool) -> None:
        self._is_active = is_active
        self.released: list[tuple[uuid.UUID, str]] = []
        self.cancelled: list[tuple[uuid.UUID, str]] = []

    def is_reservation_active(self, reservation_id: uuid.UUID) -> bool:
        return self._is_active

    def release_reservation(self, reservation_id: uuid.UUID, reason: str) -> None:
        self.released.append((reservation_id, reason))

    def cancel_order(self, order_id: uuid.UUID, reason: str) -> None:
        self.cancelled.append((order_id, reason))


class _AlwaysRejectRiskPolicy:
    def check_order_limits(
        self,
        order: OrderIntent,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> TradeDecision:
        return TradeDecision(
            approved=False,
            reason="daily_turnover_exceeded",
            available_cash=Decimal("0"),
            required_cash=Decimal("0"),
        )


def _limits() -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        effective_from=_NOW,
        max_single_name_weight=Decimal("0.10"),
        max_sector_weight=Decimal("0.30"),
        max_gross_exposure=Decimal("1.00"),
        max_daily_turnover=Decimal("0.50"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.15"),
    )


def _account() -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _intent(
    *,
    cash_reservation_id: uuid.UUID | None = None,
    side: OrderSide = OrderSide.BUY,
) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=side,
        quantity=1,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=Decimal("100"),
        cash_reservation_id=cash_reservation_id,
    )


@pytest.mark.asyncio
async def test_submit_rejects_when_broker_has_no_routing_capability() -> None:
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    intent = _intent()
    await repo.save_intent(intent)

    controller = SubmitOrdersControllerImpl(
        broker=_NoRoutingBroker(),
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
    )

    submitted = await controller.submit([intent])
    assert submitted == []
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert len(rejected) == 1
    assert "does not support order routing" in rejected[0].reason


def test_init_requires_limits_when_risk_policy_is_set() -> None:
    with pytest.raises(ValueError, match="limits must be provided"):
        SubmitOrdersControllerImpl(
            broker=_NoRoutingBroker(),
            execution_policy=_AlwaysApprovePolicy(),
            cash_engine=_CashStub(),
            event_bus=InMemoryEventBus(),
            risk_policy=_AlwaysRejectRiskPolicy(),
            limits=None,
        )


@pytest.mark.asyncio
async def test_submit_rejects_when_risk_revalidation_fails() -> None:
    """A risk policy that rejects at submit time must release the reservation
    and publish an OrderRejected with a post_approve_staleness reason."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    reservation_id = uuid.uuid4()
    intent = _intent(cash_reservation_id=reservation_id)
    await repo.save_intent(intent)

    cash = _ReservationAwareCashStub(is_active=True)

    controller = SubmitOrdersControllerImpl(
        broker=_RoutingBroker(),
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=cash,
        event_bus=bus,
        order_repo=repo,
        risk_policy=_AlwaysRejectRiskPolicy(),
        limits=_limits(),
    )

    submitted = await controller.submit([intent], account=_account())

    assert submitted == []
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert len(rejected) == 1
    assert "post_approve_staleness" in rejected[0].reason
    assert "daily_turnover_exceeded" in rejected[0].reason
    assert cash.released == [(reservation_id, "stale_approval")]


@pytest.mark.asyncio
async def test_submit_rejects_when_cash_reservation_is_stale() -> None:
    """If the buy order's reservation is no longer ACTIVE at submit time
    (e.g. TTL expired between approve and submit), the order must be
    rejected without broker placement."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    reservation_id = uuid.uuid4()
    intent = _intent(cash_reservation_id=reservation_id)
    await repo.save_intent(intent)

    cash = _ReservationAwareCashStub(is_active=False)

    controller = SubmitOrdersControllerImpl(
        broker=_RoutingBroker(),
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=cash,
        event_bus=bus,
        order_repo=repo,
    )

    submitted = await controller.submit([intent])

    assert submitted == []
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert len(rejected) == 1
    assert "cash reservation no longer active" in rejected[0].reason
    assert cash.released == [(reservation_id, "stale_approval")]


# ---------------------------------------------------------------------------
# Retry logic tests (P0-2)
# ---------------------------------------------------------------------------


class _RetryCountingBroker:
    """Broker that raises BrokerUnavailableError N times then succeeds."""

    def __init__(self, fail_count: int) -> None:
        from quant_platform.core.contracts import BrokerCapabilities

        self.calls = 0
        self.fail_count = fail_count
        self.capabilities = BrokerCapabilities(
            provider="retry_stub",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )

    async def place_order(self, order: OrderIntent) -> object:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise BrokerUnavailableError(f"transient error #{self.calls}")
        return BrokerAck(
            order_id=order.order_id,
            broker_order_id="ib-1",
            acknowledged_at=datetime.now(tz=UTC),
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        pass


class _AlwaysUnavailableBroker:
    """Broker that always raises BrokerUnavailableError."""

    def __init__(self) -> None:
        from quant_platform.core.contracts import BrokerCapabilities

        self.calls = 0
        self.capabilities = BrokerCapabilities(
            provider="always_unavailable",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )

    async def place_order(self, order: OrderIntent) -> object:
        self.calls += 1
        raise BrokerUnavailableError("always unavailable")

    async def cancel_order(self, broker_order_id: str) -> None:
        pass


class _SubmissionErrorBroker:
    """Broker that raises BrokerSubmissionError (non-retryable)."""

    def __init__(self) -> None:
        from quant_platform.core.contracts import BrokerCapabilities

        self.calls = 0
        self.capabilities = BrokerCapabilities(
            provider="rejection_stub",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )

    async def place_order(self, order: OrderIntent) -> object:
        self.calls += 1
        raise BrokerSubmissionError("broker rejected: invalid order type")

    async def cancel_order(self, broker_order_id: str) -> None:
        pass


class _AckTimeoutBroker:
    """Broker that transmitted the order but timed out waiting for ack."""

    def __init__(self) -> None:
        from quant_platform.core.contracts import BrokerCapabilities

        self.calls = 0
        self.capabilities = BrokerCapabilities(
            provider="ack_timeout_stub",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )

    async def place_order(self, order: OrderIntent) -> object:
        self.calls += 1
        raise BrokerAckTimeoutError(
            "ack timeout",
            order_id=order.order_id,
            broker_order_id="ib-uncertain-1",
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_broker_unavailable_retries_and_succeeds() -> None:
    """BrokerUnavailableError is retried up to 3 times; succeeds on 3rd attempt."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    intent = _intent()
    await repo.save_intent(intent)
    broker = _RetryCountingBroker(fail_count=2)

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
    )
    controller._retry_base_delay = 0.0  # disable sleep in tests

    submitted = await controller.submit([intent])

    assert submitted == [intent.order_id]
    assert broker.calls == 3
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert rejected == []


@pytest.mark.asyncio
async def test_submit_writes_oms_acknowledged_event_when_store_is_configured() -> None:
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    order_state = InMemoryOrderStateStore()
    intent = _intent()
    await repo.save_intent(intent)
    broker = _RetryCountingBroker(fail_count=0)

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
        order_state_store=order_state,
        clock=FakeClock(_NOW),
    )
    controller._retry_base_delay = 0.0

    submitted = await controller.submit([intent])

    assert submitted == [intent.order_id]
    events = await order_state.list_events(intent.order_id)
    assert [event.event_type for event in events] == [OrderStateEventType.ACKNOWLEDGED]
    assert events[0].broker_order_id == "ib-1"


@pytest.mark.asyncio
async def test_submit_accepts_approved_sell_without_cash_reservation() -> None:
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    intent = _intent(side=OrderSide.SELL)
    await repo.save_intent(intent)
    broker = _RetryCountingBroker(fail_count=0)

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
    )
    controller._retry_base_delay = 0.0

    submitted = await controller.submit([intent])

    assert submitted == [intent.order_id]
    assert broker.calls == 1


@pytest.mark.asyncio
async def test_broker_unavailable_exhausted_marks_terminal() -> None:
    """After max attempts, order is rejected and OrderRejected is published."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    intent = _intent()
    await repo.save_intent(intent)
    broker = _AlwaysUnavailableBroker()

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
    )
    controller._retry_base_delay = 0.0

    submitted = await controller.submit([intent])

    assert submitted == []
    assert broker.calls == 3
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert len(rejected) == 1
    assert "broker unavailable" in rejected[0].reason


@pytest.mark.asyncio
async def test_broker_submission_error_does_not_retry() -> None:
    """BrokerSubmissionError (non-retryable reject) must not trigger retries."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    intent = _intent()
    await repo.save_intent(intent)
    broker = _SubmissionErrorBroker()

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=_AlwaysApprovePolicy(),
        cash_engine=_CashStub(),
        event_bus=bus,
        order_repo=repo,
    )

    submitted = await controller.submit([intent])

    assert submitted == []
    assert broker.calls == 1
    rejected = [e for e in bus.history if isinstance(e, OrderRejected)]
    assert len(rejected) == 1
    assert "broker rejection" in rejected[0].reason


@pytest.mark.asyncio
async def test_broker_ack_timeout_preserves_order_and_reservation_state() -> None:
    """Ack timeout is an uncertain submission, not a local rejection."""
    bus = InMemoryEventBus()
    repo = InMemoryOrderRepository()
    reservation_id = uuid.uuid4()
    intent = _intent(cash_reservation_id=reservation_id)
    await repo.save_intent(intent)
    broker = _AckTimeoutBroker()
    cash = _ReservationAwareCashStub(is_active=True)
    policy = _DurableKillSwitchPolicy()

    controller = SubmitOrdersControllerImpl(
        broker=broker,
        execution_policy=policy,
        cash_engine=cash,
        event_bus=bus,
        order_repo=repo,
    )
    controller._retry_base_delay = 0.0

    submitted = await controller.submit([intent])

    assert submitted == []
    assert broker.calls == 1
    assert cash.released == []
    assert cash.cancelled == []
    assert await repo.is_terminal(intent.order_id) is False
    assert [e for e in bus.history if isinstance(e, OrderRejected)] == []
    uncertain = [e for e in bus.history if isinstance(e, OrderSubmissionUncertain)]
    assert len(uncertain) == 1
    assert uncertain[0].order_id == intent.order_id
    assert uncertain[0].broker_order_id == "ib-uncertain-1"
    kill_events = [e for e in bus.history if isinstance(e, KillSwitchActivated)]
    assert len(kill_events) == 1
    assert policy.activations == [(uncertain[0].reason, "broker_gateway")]
