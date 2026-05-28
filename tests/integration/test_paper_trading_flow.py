"""Integration test: minimal paper-trading flow.

Exercises the full approve → submit → fill → settlement → reconciliation
pipeline using the SimulatedBrokerGateway and in-memory infrastructure.

This test verifies that the concrete controllers, CashLedger, RiskPolicy,
ExecutionPolicy, and ReconciliationEngine compose correctly into a
deterministic paper-trading loop.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import (
    PlatformSettings,
    RiskSettings,
)
from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.events import (
    OrderApproved,
    OrderSubmitted,
    ReconciliationCompleted,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.support.simulated_broker import SimulatedBrokerGateway
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import (
    SettlementCalendar,
)
from quant_platform.session import Session, create_paper_session

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_INSTRUMENT = uuid.uuid4()
_RUN = uuid.uuid4()
_TARGET_ID = uuid.uuid4()

_TEST_SETTINGS = PlatformSettings(
    _env_file=None,
    risk=RiskSettings(
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.40"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.15"),
    ),
)


def _account(settled: Decimal) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=settled,
        positions=(),
    )


def _buy_intent(limit_price: Decimal, qty: int = 10) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET_ID,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=limit_price,
    )


def _make_session(initial_cash: Decimal = Decimal("50000")) -> Session:
    """Create a paper session with a FakeClock for deterministic tests."""
    clock = FakeClock(_NOW)
    return create_paper_session(
        _TEST_SETTINGS,
        initial_cash=initial_cash,
        strategy_run_id=_RUN,
        clock=clock,
    )


class TestPaperTradingFlow:
    @pytest.mark.asyncio
    async def test_approve_submit_reconcile(self) -> None:
        """Full pipeline: approve → submit → reconcile with no discrepancies."""
        initial_cash = Decimal("50000")
        s = _make_session(initial_cash)

        broker: SimulatedBrokerGateway = s.broker  # type: ignore[assignment]
        broker.set_market_price(_INSTRUMENT, Decimal("100"))
        await broker.connect()

        account = _account(initial_cash)

        # --- Step 1: Approve ---
        intent = _buy_intent(Decimal("100"), qty=10)
        approved, rejected = await s.approve_ctrl.approve([intent], account)
        assert len(approved) == 1
        assert len(rejected) == 0
        assert approved[0].cash_reservation_id is not None
        assert s.cash_engine.reserved_cash > Decimal("0")

        approval_events = [e for e in s.event_bus.history if isinstance(e, OrderApproved)]
        assert len(approval_events) == 1

        # --- Step 2: Submit ---
        submitted = await s.submit_ctrl.submit(approved)
        assert len(submitted) == 1

        submit_events = [e for e in s.event_bus.history if isinstance(e, OrderSubmitted)]
        assert len(submit_events) == 1

        # --- Step 3: Process fill (simulated broker fills immediately) ---
        events = await broker.drain_lifecycle_events()
        fills = [event.fill for event in events if isinstance(event, BrokerFillEvent)]
        assert len(fills) == 1
        fill = fills[0]
        s.cash_engine.apply_fill(fill, is_order_complete=True)
        assert s.cash_engine.reserved_cash == Decimal("0")

        # --- Step 4: Reconcile ---
        broker_account = await broker.sync_account()
        await s.position_repo.save_snapshot(broker_account)

        await s.recon_ctrl.reconcile(_RUN)
        recon_events = [e for e in s.event_bus.history if isinstance(e, ReconciliationCompleted)]
        assert len(recon_events) >= 1
        assert not recon_events[-1].requires_operator_action
        assert not s.execution_policy.kill_switch_active

    @pytest.mark.asyncio
    async def test_insufficient_cash_rejected(self) -> None:
        """An order that exceeds settled cash must be rejected at approval."""
        initial_cash = Decimal("500")
        s = _make_session(initial_cash)
        account = _account(initial_cash)

        intent = _buy_intent(Decimal("100"), qty=10)  # $1000 notional > $500
        approved, rejected = await s.approve_ctrl.approve([intent], account)
        assert len(approved) == 0
        assert len(rejected) == 1

    @pytest.mark.asyncio
    async def test_sell_settlement_flow(self) -> None:
        """Sell fill → settlement projection → settle lot → cash restored."""
        initial_cash = Decimal("50000")
        clock = FakeClock(_NOW)
        cal = SettlementCalendar()
        account = _account(initial_cash)

        broker = SimulatedBrokerGateway(clock=clock, initial_cash=initial_cash)
        broker.set_market_price(_INSTRUMENT, Decimal("100"))
        broker._positions[_INSTRUMENT] = 100
        broker._avg_costs[_INSTRUMENT] = Decimal("90")
        await broker.connect()

        ledger = CashLedger(clock=clock, settlement_calendar=cal, initial_snapshot=account)

        sell = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=_RUN,
            portfolio_target_id=_TARGET_ID,
            instrument_id=_INSTRUMENT,
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            created_at=_NOW,
            limit_price=Decimal("100"),
        )
        await broker.place_order(sell)
        events = await broker.drain_lifecycle_events()
        fills = [event.fill for event in events if isinstance(event, BrokerFillEvent)]
        assert len(fills) == 1

        fill = fills[0]
        ledger.apply_fill(fill)
        assert ledger.unsettled_cash > Decimal("0")

        lots = ledger.project_settlement([fill])
        assert len(lots) == 1

        clock.advance(days=1)
        ledger.settle_lot(lots[0])
        assert ledger.unsettled_cash == Decimal("0")
        assert ledger.settled_cash > initial_cash
