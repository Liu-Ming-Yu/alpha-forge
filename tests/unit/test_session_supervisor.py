"""Unit tests for BrokerSessionSupervisor."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.config import BrokerSettings
from quant_platform.core.contracts import (
    BrokerCapabilities,
    BrokerHealth,
    BrokerHealthStatus,
)
from quant_platform.core.domain.orders import (
    BrokerOrder,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.events import BrokerSessionHealthChanged, KillSwitchActivated
from quant_platform.infrastructure.event_bus import InMemoryEventBus
from quant_platform.infrastructure.repositories import InMemoryOrderRepository
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.execution_service.session.session_supervisor import (
    BrokerSessionSupervisor,
)

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)


class _Clock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def advance(self, seconds: int) -> None:
        self._now += timedelta(seconds=seconds)


class _StubGateway:
    def __init__(self) -> None:
        self.capabilities = BrokerCapabilities(
            provider="stub",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )
        self.connected = False
        self.connect_calls = 0
        self.fail_connect = False
        self.health_status = BrokerHealthStatus.CONNECTED
        self.open_orders: list[BrokerOrder] = []
        self.cancelled: list[str] = []
        self.snapshot = AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_NOW,
            settled_cash=Decimal("10000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("10000"),
            net_asset_value=Decimal("10000"),
            positions=(),
        )

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def health_check(self) -> BrokerHealth:
        return BrokerHealth(
            status=self.health_status,
            latency_ms=1,
            last_heartbeat_at=_NOW,
            detail="stub",
        )

    async def sync_account(self) -> AccountSnapshot:
        return self.snapshot

    async def sync_positions(self) -> list:
        return []

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    async def place_order(self, order: OrderIntent) -> object:
        raise NotImplementedError

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)


class _StubReconcile:
    def __init__(self) -> None:
        self.calls = 0
        self.raise_on_reconcile = False

    async def reconcile(self, strategy_run_id: uuid.UUID) -> None:
        self.calls += 1
        if self.raise_on_reconcile:
            raise RuntimeError("reconcile failed")


def _intent(order_id: uuid.UUID, tif: TimeInForce) -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        time_in_force=tif,
        created_at=_NOW,
        limit_price=Decimal("100"),
    )


@pytest.mark.asyncio
async def test_startup_recovery_success_publishes_health_transition() -> None:
    clock = _Clock(_NOW)
    gateway = _StubGateway()
    bus = InMemoryEventBus()
    throttle = OrderThrottle(clock)
    reconcile = _StubReconcile()
    repo = InMemoryOrderRepository()

    supervisor = BrokerSessionSupervisor(
        session_gateway=gateway,
        order_gateway=gateway,
        event_bus=bus,
        execution_policy=throttle,
        reconcile_controller=reconcile,
        order_repo=repo,
        clock=clock,
        settings=BrokerSettings(
            heartbeat_interval_seconds=0.0,
            max_consecutive_health_failures=1,
            reconnect_base_delay=0.0,
            reconnect_max_delay=0.0,
        ),
    )

    ok = await supervisor.startup_recovery(uuid.uuid4())
    assert ok
    assert gateway.connect_calls >= 1
    assert reconcile.calls == 1
    assert any(isinstance(e, BrokerSessionHealthChanged) for e in bus.history)


@pytest.mark.asyncio
async def test_reconnect_failure_triggers_kill_switch() -> None:
    clock = _Clock(_NOW)
    gateway = _StubGateway()
    gateway.health_status = BrokerHealthStatus.DISCONNECTED
    gateway.fail_connect = True
    bus = InMemoryEventBus()
    throttle = OrderThrottle(clock)
    reconcile = _StubReconcile()
    repo = InMemoryOrderRepository()

    supervisor = BrokerSessionSupervisor(
        session_gateway=gateway,
        order_gateway=gateway,
        event_bus=bus,
        execution_policy=throttle,
        reconcile_controller=reconcile,
        order_repo=repo,
        clock=clock,
        settings=BrokerSettings(
            heartbeat_interval_seconds=0.0,
            max_consecutive_health_failures=1,
            reconnect_base_delay=0.0,
            reconnect_max_delay=0.0,
        ),
    )

    healthy = await supervisor.poll_health(uuid.uuid4())
    assert not healthy
    assert throttle.kill_switch_active
    assert any(isinstance(e, KillSwitchActivated) for e in bus.history)


@pytest.mark.asyncio
async def test_failed_reconcile_after_reconnect_activates_kill_switch() -> None:
    clock = _Clock(_NOW)
    gateway = _StubGateway()
    gateway.health_status = BrokerHealthStatus.DISCONNECTED
    bus = InMemoryEventBus()
    throttle = OrderThrottle(clock)
    reconcile = _StubReconcile()
    reconcile.raise_on_reconcile = True
    repo = InMemoryOrderRepository()

    supervisor = BrokerSessionSupervisor(
        session_gateway=gateway,
        order_gateway=gateway,
        event_bus=bus,
        execution_policy=throttle,
        reconcile_controller=reconcile,
        order_repo=repo,
        clock=clock,
        settings=BrokerSettings(
            heartbeat_interval_seconds=0.0,
            max_consecutive_health_failures=2,
            reconnect_base_delay=0.0,
            reconnect_max_delay=0.0,
        ),
    )

    healthy = await supervisor.poll_health(uuid.uuid4())
    assert not healthy
    assert throttle.kill_switch_active
    kill_events = [e for e in bus.history if isinstance(e, KillSwitchActivated)]
    assert len(kill_events) >= 1
    assert "reconcile after reconnect failed" in kill_events[-1].reason


@pytest.mark.asyncio
async def test_cleanup_stale_day_orders_cancels_only_day_intents() -> None:
    clock = _Clock(_NOW)
    gateway = _StubGateway()
    bus = InMemoryEventBus()
    throttle = OrderThrottle(clock)
    reconcile = _StubReconcile()
    repo = InMemoryOrderRepository()

    stale_day_order_id = uuid.uuid4()
    stale_gtc_order_id = uuid.uuid4()
    await repo.save_intent(_intent(stale_day_order_id, TimeInForce.DAY))
    await repo.save_intent(_intent(stale_gtc_order_id, TimeInForce.GTC))

    stale_time = _NOW - timedelta(minutes=31)
    gateway.open_orders = [
        BrokerOrder(
            order_id=stale_day_order_id,
            status=OrderStatus.SUBMITTED,
            last_updated_at=stale_time,
            broker_order_id="2001",
        ),
        BrokerOrder(
            order_id=stale_gtc_order_id,
            status=OrderStatus.SUBMITTED,
            last_updated_at=stale_time,
            broker_order_id="2002",
        ),
    ]

    supervisor = BrokerSessionSupervisor(
        session_gateway=gateway,
        order_gateway=gateway,
        event_bus=bus,
        execution_policy=throttle,
        reconcile_controller=reconcile,
        order_repo=repo,
        clock=clock,
        settings=BrokerSettings(
            stale_day_order_cleanup_minutes=30,
            heartbeat_interval_seconds=0.0,
            max_consecutive_health_failures=1,
            reconnect_base_delay=0.0,
            reconnect_max_delay=0.0,
        ),
    )

    cancelled = await supervisor.cleanup_stale_day_orders(uuid.uuid4())
    assert cancelled == 1
    assert gateway.cancelled == ["2001"]
