from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.core.contracts import BrokerAck, BrokerCapabilities
from quant_platform.core.domain.orders import (
    BrokerOrder,
    OrderIntent,
    OrderSide,
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import BrokerOrderCancelled
from quant_platform.core.domain.production import ExecutionTacticPolicy
from quant_platform.engines.session.cycle_steps import apply_cycle_passive_reprice
from quant_platform.engines.session.passive_reprice import run_passive_reprice_once
from quant_platform.infrastructure.repositories import InMemoryOrderRepository
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.v2.state import InMemoryOrderStateStore
from quant_platform.services.execution_service.orders.router import DefaultExecutionRouter
from quant_platform.services.execution_service.passive_reprice import (
    PassiveLimitRepriceCoordinator,
)

_NOW = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)


class _Broker:
    def __init__(self, clock: FakeClock, open_orders: list[BrokerOrder]) -> None:
        self._clock = clock
        self.open_orders = open_orders
        self.cancelled: list[str] = []
        self.placed: list[OrderIntent] = []
        self.fail_cancel = False
        self.fail_place = False
        self.capabilities = BrokerCapabilities(
            provider="stub",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=False,
        )

    async def fetch_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    async def cancel_order(self, broker_order_id: str) -> None:
        if self.fail_cancel:
            raise RuntimeError("cancel unavailable")
        self.cancelled.append(broker_order_id)
        self.open_orders = [
            order for order in self.open_orders if order.broker_order_id != broker_order_id
        ]

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        if self.fail_place:
            raise RuntimeError("place unavailable")
        self.placed.append(order)
        ack = BrokerAck(
            order_id=order.order_id,
            broker_order_id=f"replacement-{len(self.placed)}",
            acknowledged_at=self._clock.now(),
        )
        self.open_orders.append(
            BrokerOrder(
                order_id=order.order_id,
                status=OrderStatus.SUBMITTED,
                broker_order_id=ack.broker_order_id,
                submitted_at=self._clock.now(),
                last_updated_at=self._clock.now(),
            )
        )
        return ack


class _LifecycleBroker(_Broker):
    def __init__(self, clock: FakeClock, open_orders: list[BrokerOrder]) -> None:
        super().__init__(clock, open_orders)
        self.lifecycle_events: list[BrokerOrderCancelled] = []

    async def cancel_order(self, broker_order_id: str) -> None:
        order_id = next(
            order.order_id for order in self.open_orders if order.broker_order_id == broker_order_id
        )
        await super().cancel_order(broker_order_id)
        self.lifecycle_events.append(
            BrokerOrderCancelled(
                order_id=order_id,
                broker_order_id=broker_order_id,
                reason="operator cancel",
                occurred_at=self._clock.now(),
            )
        )

    async def drain_lifecycle_events(self) -> list[BrokerOrderCancelled]:
        events = list(self.lifecycle_events)
        self.lifecycle_events.clear()
        return events


def _intent(
    *,
    created_at: datetime,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    limit_price: Decimal | None = Decimal("100"),
) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=side,
        quantity=100,
        order_type=order_type,
        time_in_force=TimeInForce.DAY,
        created_at=created_at,
        limit_price=limit_price,
    )


def _broker_order(intent: OrderIntent, *, submitted_at: datetime | None = None) -> BrokerOrder:
    return BrokerOrder(
        order_id=intent.order_id,
        status=OrderStatus.SUBMITTED,
        broker_order_id="ib-1",
        submitted_at=submitted_at,
        last_updated_at=submitted_at or intent.created_at,
    )


async def _repo_with(intent: OrderIntent) -> InMemoryOrderRepository:
    repo = InMemoryOrderRepository()
    await repo.save_intent(intent)
    return repo


def _policy(**overrides: object) -> ExecutionTacticPolicy:
    params = {
        "passive_limit_enabled": True,
        "reprice_interval_seconds": 300,
        "max_reprices_per_order": 3,
        "min_reprice_improvement_bps": 5.0,
        "adverse_drift_escalate_bps": 25.0,
    }
    params.update(overrides)
    return ExecutionTacticPolicy(**params)


@pytest.mark.asyncio
async def test_passive_reprice_skips_young_order_and_records_evidence() -> None:
    clock = FakeClock(_NOW)
    intent = _intent(created_at=_NOW)
    order_state = InMemoryOrderStateStore()
    broker = _Broker(clock, [_broker_order(intent, submitted_at=_NOW)])
    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(),
        router=DefaultExecutionRouter(_policy(), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        order_state=order_state,
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("101"),
    )

    decisions = await coordinator.run_once()

    assert [(decision.action, decision.reason) for decision in decisions] == [
        ("skipped", "not_due")
    ]
    assert broker.cancelled == []
    events = await order_state.list_events(intent.order_id)
    assert events[0].event_type == OrderStateEventType.ROUTED
    assert events[0].payload["action"] == "skipped"


@pytest.mark.asyncio
async def test_passive_reprice_cancels_due_order_without_replacement_factory() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    order_state = InMemoryOrderStateStore()
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])
    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(),
        router=DefaultExecutionRouter(_policy(), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        order_state=order_state,
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("99"),
    )

    decisions = await coordinator.run_once()

    assert decisions[0].action == "cancelled"
    assert decisions[0].new_limit_price == Decimal("99")
    assert broker.cancelled == ["ib-1"]
    assert broker.placed == []
    events = await order_state.list_events(intent.order_id)
    assert events[0].event_type == OrderStateEventType.CANCEL_REQUESTED
    assert events[0].payload["reason"] == "cancelled_without_replacement_factory"


@pytest.mark.asyncio
async def test_passive_reprice_places_replacement_from_factory() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    order_state = InMemoryOrderStateStore()
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])

    def _replacement(
        original: OrderIntent,
        *,
        new_limit_price: Decimal,
        route: object,
        requested_at: datetime,
    ) -> OrderIntent:
        _ = route
        return OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=original.strategy_run_id,
            portfolio_target_id=original.portfolio_target_id,
            instrument_id=original.instrument_id,
            side=original.side,
            quantity=original.quantity,
            order_type=original.order_type,
            time_in_force=original.time_in_force,
            created_at=requested_at,
            limit_price=new_limit_price,
            cash_reservation_id=original.cash_reservation_id,
        )

    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(),
        router=DefaultExecutionRouter(_policy(), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        order_state=order_state,
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("99"),
        replacement_factory=_replacement,
    )

    decisions = await coordinator.run_once()

    assert decisions[0].action == "replaced"
    assert decisions[0].replacement_order_id == broker.placed[0].order_id
    assert broker.cancelled == ["ib-1"]
    assert broker.placed[0].limit_price == Decimal("99")


@pytest.mark.asyncio
async def test_passive_reprice_escalates_after_adverse_drift_threshold() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at, side=OrderSide.BUY, limit_price=Decimal("100"))
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])
    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(adverse_drift_escalate_bps=10.0),
        router=DefaultExecutionRouter(_policy(adverse_drift_escalate_bps=10.0), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        order_state=InMemoryOrderStateStore(),
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("100.20"),
    )

    decisions = await coordinator.run_once()

    assert decisions[0].action == "escalated"
    assert broker.cancelled == ["ib-1"]


@pytest.mark.asyncio
async def test_passive_reprice_refuses_after_max_reprices() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    order_state = InMemoryOrderStateStore()
    await order_state.append(
        # Existing evidence from a prior coordinator run.
        OrderStateEvent(
            event_id=uuid.uuid4(),
            order_id=intent.order_id,
            event_type=OrderStateEventType.CANCEL_REQUESTED,
            occurred_at=created_at,
            status=OrderStatus.SUBMITTED,
            broker_order_id="ib-0",
            payload={"source": "passive_reprice", "action": "cancelled"},
        )
    )
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])
    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(max_reprices_per_order=1),
        router=DefaultExecutionRouter(_policy(max_reprices_per_order=1), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        order_state=order_state,
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("99"),
    )

    decisions = await coordinator.run_once()

    assert [(decision.action, decision.reason) for decision in decisions] == [
        ("skipped", "max_reprices_exceeded")
    ]
    assert broker.cancelled == []


@pytest.mark.asyncio
async def test_passive_reprice_fails_closed_when_cancel_fails() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])
    broker.fail_cancel = True
    coordinator = PassiveLimitRepriceCoordinator(
        policy=_policy(),
        router=DefaultExecutionRouter(_policy(), broker=broker),
        broker=broker,
        order_repo=await _repo_with(intent),
        clock=clock,
        reference_price_lookup=lambda _intent: Decimal("99"),
        replacement_factory=lambda original, **kwargs: original,
    )

    decisions = await coordinator.run_once()

    assert decisions[0].action == "failed"
    assert "cancel_failed" in decisions[0].reason
    assert broker.placed == []


@pytest.mark.asyncio
async def test_runtime_helper_skips_missing_reference_price_without_cancelling() -> None:
    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    broker = _Broker(clock, [_broker_order(intent, submitted_at=created_at)])
    order_state = InMemoryOrderStateStore()
    session = SimpleNamespace(
        execution_tactic_policy=_policy(),
        trading_broker=broker,
        broker=broker,
        order_repo=await _repo_with(intent),
        v2_order_state=order_state,
        clock=clock,
    )

    decisions = await run_passive_reprice_once(session=session, market_prices={})

    assert [(decision.action, decision.reason) for decision in decisions] == [
        ("skipped", "missing_reference_price")
    ]
    assert broker.cancelled == []
    events = await order_state.list_events(intent.order_id)
    assert events[0].payload["action"] == "skipped"


@pytest.mark.asyncio
async def test_cycle_passive_reprice_drains_cancellation_lifecycle() -> None:
    class _Coordinator:
        def __init__(self) -> None:
            self.events: list[BrokerOrderCancelled] = []

        async def process_lifecycle_events(
            self,
            events: list[BrokerOrderCancelled],
        ) -> None:
            self.events.extend(events)

    clock = FakeClock(_NOW)
    created_at = datetime(2026, 5, 8, 14, 0, tzinfo=UTC)
    intent = _intent(created_at=created_at)
    broker = _LifecycleBroker(clock, [_broker_order(intent, submitted_at=created_at)])
    coordinator = _Coordinator()
    session = SimpleNamespace(
        execution_tactic_policy=_policy(),
        trading_broker=broker,
        broker=broker,
        order_repo=await _repo_with(intent),
        v2_order_state=InMemoryOrderStateStore(),
        clock=clock,
        lifecycle_feed=broker,
        coordinator=coordinator,
    )

    await apply_cycle_passive_reprice(
        session=session,
        market_prices={intent.instrument_id: Decimal("99")},
        engine_name="test_engine",
    )

    assert broker.cancelled == ["ib-1"]
    assert [event.order_id for event in coordinator.events] == [intent.order_id]
