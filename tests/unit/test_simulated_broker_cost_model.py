"""Unit tests for SimulatedBrokerGateway execution-cost hooks."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.support.simulated_broker import (
    ParticipationFillModel,
    SimulatedBrokerGateway,
    SimulatedLiquidityProfile,
)

_UTC = UTC


def _intent(
    *,
    now: datetime,
    instrument_id: uuid.UUID,
    side: OrderSide = OrderSide.BUY,
    quantity: int = 100,
    limit_price: Decimal = Decimal("100"),
) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=now,
        limit_price=limit_price,
    )


@pytest.mark.asyncio
async def test_configured_cost_model_adjusts_fill_price_and_commission() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    clock = FakeClock(now)
    broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("10000"))
    instrument_id = uuid.uuid4()

    def _adjust(order: OrderIntent, reference_price: Decimal) -> Decimal:
        _ = order
        return reference_price * Decimal("1.01")

    def _commission(order: OrderIntent, fill_price: Decimal) -> Decimal:
        _ = order
        _ = fill_price
        return Decimal("2.50")

    broker.configure_execution_cost_model(
        fill_price_adjuster=_adjust,
        commission_calculator=_commission,
    )
    await broker.connect()

    intent = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=now,
        limit_price=Decimal("100"),
    )
    await broker.place_order(intent)
    events = await broker.drain_lifecycle_events()
    fills = [event.fill for event in events if isinstance(event, BrokerFillEvent)]
    assert len(fills) == 1
    assert fills[0].fill_price == Decimal("101.00")
    assert fills[0].commission == Decimal("2.50")

    account = await broker.sync_account()
    expected_cash = Decimal("10000") - Decimal("10") * Decimal("101.00") - Decimal("2.50")
    assert account.settled_cash == expected_cash


def test_participation_fill_model_caps_partial_fill_and_costs() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    instrument_id = uuid.uuid4()
    model = ParticipationFillModel(
        liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
            adv_shares_20d=1_000.0,
            spread_bps=8.0,
        ),
        max_participation_pct=0.05,
        fill_price_adjuster=lambda _order, price, _qty, _liq: price * Decimal("1.01"),
        commission_calculator=lambda _order, _price, qty: Decimal(str(qty)) * Decimal("0.005"),
    )

    plan = model.plan(
        _intent(now=now, instrument_id=instrument_id, quantity=100),
        Decimal("100"),
    )

    assert plan.requested_quantity == 100
    assert plan.filled_quantity == 50
    assert plan.is_complete is False
    assert plan.participation_pct == 0.05
    assert plan.fill_price == Decimal("101.00")
    assert plan.commission == Decimal("0.250")
    assert plan.implementation_shortfall_bps == pytest.approx(100.0)


def test_participation_fill_model_full_fill_and_missing_adv_fallback() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    instrument_id = uuid.uuid4()
    model = ParticipationFillModel(
        liquidity_lookup=lambda _inst, _price: None,
        max_participation_pct=0.05,
        fallback_adv_shares=10_000.0,
        fallback_spread_bps=10.0,
    )

    plan = model.plan(
        _intent(now=now, instrument_id=instrument_id, quantity=1_000),
        Decimal("50"),
    )

    assert plan.filled_quantity == 1_000
    assert plan.is_complete is True
    assert plan.adv_shares_20d == 20_000.0
    assert plan.spread_bps == 10.0

    zero_adv_model = ParticipationFillModel(
        liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
            adv_shares_20d=0.0,
            spread_bps=25.0,
        ),
        max_participation_pct=0.05,
        fallback_adv_shares=10_000.0,
        fallback_spread_bps=10.0,
    )
    zero_adv_plan = zero_adv_model.plan(
        _intent(now=now, instrument_id=instrument_id, quantity=1_000),
        Decimal("50"),
    )

    assert zero_adv_plan.filled_quantity == 1_000
    assert zero_adv_plan.is_complete is True
    assert zero_adv_plan.adv_shares_20d == 20_000.0


def test_participation_fill_model_sell_shortfall_direction() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    instrument_id = uuid.uuid4()
    model = ParticipationFillModel(
        liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
            adv_shares_20d=1_000_000.0,
            spread_bps=2.0,
        ),
        max_participation_pct=0.05,
        fill_price_adjuster=lambda _order, price, _qty, _liq: price * Decimal("0.99"),
    )

    buy_plan = model.plan(
        _intent(now=now, instrument_id=instrument_id, side=OrderSide.BUY),
        Decimal("100"),
    )
    sell_plan = model.plan(
        _intent(now=now, instrument_id=instrument_id, side=OrderSide.SELL),
        Decimal("100"),
    )

    assert buy_plan.implementation_shortfall_bps == pytest.approx(-100.0)
    assert sell_plan.implementation_shortfall_bps == pytest.approx(100.0)


def test_participation_fill_model_applies_stale_price_and_close_auction_spread() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    instrument_id = uuid.uuid4()
    order = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MOC,
        time_in_force=TimeInForce.DAY,
        created_at=now,
    )
    model = ParticipationFillModel(
        liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
            adv_shares_20d=10_000.0,
            spread_bps=4.0,
        ),
        max_participation_pct=0.05,
        stale_price_bps=10.0,
        close_auction_spread_multiplier=2.0,
    )

    plan = model.plan(order, Decimal("100"))

    assert plan.fill_price == Decimal("100.100")
    assert plan.spread_bps == pytest.approx(8.0)
    assert plan.implementation_shortfall_bps == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_execution_model_partial_fill_stays_open() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    clock = FakeClock(now)
    broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("10000"))
    instrument_id = uuid.uuid4()
    broker.configure_execution_model(
        ParticipationFillModel(
            liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
                adv_shares_20d=1_000.0,
                spread_bps=8.0,
            ),
            max_participation_pct=0.05,
        )
    )
    await broker.connect()

    intent = _intent(now=now, instrument_id=instrument_id, quantity=100)
    await broker.place_order(intent)
    events = await broker.drain_lifecycle_events()
    open_orders = await broker.fetch_open_orders()

    assert len(events) == 1
    assert isinstance(events[0], BrokerFillEvent)
    assert events[0].fill.quantity == 50
    assert events[0].fill.broker_execution_id is not None
    assert events[0].fill.broker_execution_id.endswith("-1000-1")
    assert events[0].is_complete is False
    assert [order.order_id for order in open_orders] == [intent.order_id]


@pytest.mark.asyncio
async def test_simulated_manual_partial_fills_get_distinct_execution_ids() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    clock = FakeClock(now)
    broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("10000"))
    instrument_id = uuid.uuid4()
    broker.configure_execution_model(
        ParticipationFillModel(
            liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
                adv_shares_20d=1_000.0,
                spread_bps=8.0,
            ),
            max_participation_pct=0.05,
        )
    )
    await broker.connect()

    intent = _intent(now=now, instrument_id=instrument_id, quantity=100)
    ack = await broker.place_order(intent)
    first = broker.simulate_partial_fill(
        intent.order_id,
        quantity=25,
        price=Decimal("100"),
        is_complete=False,
    )
    second = broker.simulate_partial_fill(
        intent.order_id,
        quantity=25,
        price=Decimal("100"),
        is_complete=True,
    )

    assert first.broker_order_id == ack.broker_order_id
    assert second.broker_order_id == ack.broker_order_id
    assert first.broker_execution_id != second.broker_execution_id


@pytest.mark.asyncio
async def test_execution_model_complete_fill_closes_order() -> None:
    now = datetime(2026, 4, 13, 14, 0, tzinfo=_UTC)
    clock = FakeClock(now)
    broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("10000"))
    instrument_id = uuid.uuid4()
    broker.configure_execution_model(
        ParticipationFillModel(
            liquidity_lookup=lambda _inst, _price: SimulatedLiquidityProfile(
                adv_shares_20d=1_000_000.0,
                spread_bps=2.0,
            ),
            max_participation_pct=0.05,
        )
    )
    await broker.connect()

    await broker.place_order(_intent(now=now, instrument_id=instrument_id, quantity=100))
    events = await broker.drain_lifecycle_events()
    open_orders = await broker.fetch_open_orders()

    assert len(events) == 1
    assert isinstance(events[0], BrokerFillEvent)
    assert events[0].fill.quantity == 100
    assert events[0].is_complete is True
    assert open_orders == []
