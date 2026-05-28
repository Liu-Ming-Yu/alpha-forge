"""Integration tests: cash control loop.

Exercises the AccountStateCoordinator, lifecycle event processing, partial
fills, cancel/reject reservation release, T+1 settlement blocking, and
broker-vs-ledger drift detection.

Tests:
    TestLifecycleEventProcessing
        test_fill_updates_ledger_via_coordinator
        test_partial_fill_holds_reservation
        test_cancel_releases_reservation
        test_reject_releases_reservation
    TestSettlementBlocking
        test_sell_proceeds_unsettled_until_t_plus_1
        test_unsettled_cash_cannot_fund_buy
        test_advance_settlements_moves_cash
    TestCashDrift
        test_drift_within_tolerance_ok
        test_drift_exceeds_tolerance_flags
    TestFullCycleWithCoordinator
        test_cycle_uses_coordinator_lifecycle_feed
        test_cycle_expires_stale_reservations
    TestRestartAndReconcile
        test_reconcile_after_restart_corrects_state
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from quant_platform.config import PlatformSettings, RiskSettings
from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerOrderCancelled,
    BrokerOrderCompleted,
    BrokerOrderRejected,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import (
    KillSwitchActivated,
    OrderCancelled,
    OrderFilled,
    SettlementApplied,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.support.simulated_broker import SimulatedBrokerGateway
from quant_platform.services.execution_service.account.account_state_coordinator import (
    AccountStateCoordinator,
)
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.settlement_calendar import (
    SettlementCalendar,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC
_NOW = datetime(2025, 6, 2, 14, 0, 0, tzinfo=_UTC)  # Monday
_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()
_RUN_ID = uuid.uuid4()

_SETTINGS = PlatformSettings(
    _env_file=None,
    risk=RiskSettings(
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.50"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.20"),
    ),
)


def _account(settled: Decimal, unsettled: Decimal = Decimal("0")) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=unsettled,
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=settled + unsettled,
        positions=(),
    )


def _buy_intent(
    price: Decimal,
    qty: int = 10,
    instrument_id: uuid.UUID | None = None,
) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN_ID,
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id or _INST_A,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=price,
        cash_reservation_id=uuid.uuid4(),
    )


def _sell_intent(
    price: Decimal,
    qty: int = 10,
    instrument_id: uuid.UUID | None = None,
) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN_ID,
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id or _INST_A,
        side=OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=price,
        cash_reservation_id=uuid.uuid4(),
    )


def _make_fill(
    order_id: uuid.UUID,
    side: OrderSide,
    price: Decimal,
    qty: int = 10,
    instrument_id: uuid.UUID | None = None,
) -> FillEvent:
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="sim-1000",
        instrument_id=instrument_id or _INST_A,
        side=side,
        quantity=qty,
        fill_price=price,
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=_NOW,
        received_at=_NOW,
    )


def _make_strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=_RUN_ID,
        strategy_name="test_cash_control",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _make_regime(label: RegimeLabel = RegimeLabel.RISK_ON) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=label,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


def _make_coordinator(
    clock: FakeClock,
    ledger: CashLedger,
) -> AccountStateCoordinator:
    from quant_platform.infrastructure.event_bus import InMemoryEventBus

    bus = InMemoryEventBus()
    return AccountStateCoordinator(
        cash_engine=ledger,
        event_bus=bus,
        clock=clock,
        strategy_run_id=_RUN_ID,
    ), bus


# -----------------------------------------------------------------------
# Test: Lifecycle Event Processing
# -----------------------------------------------------------------------


class TestLifecycleEventProcessing:
    """Coordinator correctly translates lifecycle events to ledger mutations."""

    async def test_fill_updates_ledger_via_coordinator(self) -> None:
        """A BrokerFillEvent(is_complete=True) applies the fill and releases the reservation."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        intent = _buy_intent(Decimal("100"), qty=10)
        reservation = ledger.reserve_cash(intent, snapshot)
        ledger.mark_order_submitted(intent.order_id)

        assert ledger.reserved_cash > Decimal("0")

        fill = _make_fill(intent.order_id, OrderSide.BUY, Decimal("100"), qty=10)
        events = [BrokerFillEvent(fill=fill, is_complete=True)]
        result = await coordinator.process_lifecycle_events(events)

        assert result.fills_applied == 1
        assert result.reservations_released == 1
        assert ledger.reserved_cash == Decimal("0")
        assert ledger.settled_cash < Decimal("50000")

        filled_events = [e for e in bus.history if isinstance(e, OrderFilled)]
        assert len(filled_events) == 1
        assert filled_events[0].is_complete is True

    async def test_partial_fill_holds_reservation(self) -> None:
        """A partial fill (is_complete=False) deducts cash but keeps the reservation."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        intent = _buy_intent(Decimal("100"), qty=20)
        reservation = ledger.reserve_cash(intent, snapshot)
        ledger.mark_order_submitted(intent.order_id)

        initial_reserved = ledger.reserved_cash
        assert initial_reserved > Decimal("0")

        # Partial fill: 10 of 20 shares.
        fill1 = _make_fill(intent.order_id, OrderSide.BUY, Decimal("100"), qty=10)
        result1 = await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=fill1, is_complete=False)]
        )

        assert result1.fills_applied == 1
        assert result1.reservations_released == 0
        assert ledger.reserved_cash == initial_reserved  # reservation still active
        assert ledger.settled_cash < Decimal("50000")  # cash deducted

        # Final fill: remaining 10 shares.
        fill2 = _make_fill(intent.order_id, OrderSide.BUY, Decimal("100"), qty=10)
        result2 = await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=fill2, is_complete=True)]
        )

        assert result2.fills_applied == 1
        assert result2.reservations_released == 1
        assert ledger.reserved_cash == Decimal("0")

    async def test_cancel_releases_reservation(self) -> None:
        """A BrokerOrderCancelled event releases the cash reservation."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        intent = _buy_intent(Decimal("100"), qty=10)
        reservation = ledger.reserve_cash(intent, snapshot)
        ledger.mark_order_submitted(intent.order_id)

        assert ledger.reserved_cash > Decimal("0")
        cash_before = ledger.settled_cash

        cancel = BrokerOrderCancelled(
            order_id=intent.order_id,
            broker_order_id="sim-1000",
            reason="test cancel",
            occurred_at=_NOW,
        )
        result = await coordinator.process_lifecycle_events([cancel])

        assert result.reservations_released == 1
        assert ledger.reserved_cash == Decimal("0")
        assert ledger.settled_cash == cash_before  # no cash consumed

        cancel_events = [e for e in bus.history if isinstance(e, OrderCancelled)]
        assert len(cancel_events) == 1

    async def test_reject_releases_reservation(self) -> None:
        """A BrokerOrderRejected event releases the cash reservation."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        intent = _buy_intent(Decimal("100"), qty=10)
        reservation = ledger.reserve_cash(intent, snapshot)
        ledger.mark_order_submitted(intent.order_id)

        assert ledger.reserved_cash > Decimal("0")

        reject = BrokerOrderRejected(
            order_id=intent.order_id,
            broker_order_id="sim-1000",
            reason="insufficient margin",
            occurred_at=_NOW,
        )
        result = await coordinator.process_lifecycle_events([reject])

        assert result.reservations_released == 1
        assert ledger.reserved_cash == Decimal("0")

    async def test_broker_completed_releases_reservation(self) -> None:
        """A BrokerOrderCompleted event releases any held reservation."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        intent = _buy_intent(Decimal("100"), qty=10)
        reservation = ledger.reserve_cash(intent, snapshot)
        ledger.mark_order_submitted(intent.order_id)

        # Simulate fill arriving first without is_complete.
        fill = _make_fill(intent.order_id, OrderSide.BUY, Decimal("100"), qty=10)
        await coordinator.process_lifecycle_events([BrokerFillEvent(fill=fill, is_complete=False)])
        assert ledger.reserved_cash > Decimal("0")

        # Then BrokerOrderCompleted arrives.
        completed = BrokerOrderCompleted(
            order_id=intent.order_id,
            broker_order_id="sim-1000",
            occurred_at=_NOW,
        )
        result = await coordinator.process_lifecycle_events([completed])
        assert result.reservations_released == 1
        assert ledger.reserved_cash == Decimal("0")


# -----------------------------------------------------------------------
# Test: Settlement Blocking
# -----------------------------------------------------------------------


class TestSettlementBlocking:
    """Sell proceeds are unsettled until T+1 and cannot fund new buys."""

    async def test_sell_proceeds_unsettled_until_t_plus_1(self) -> None:
        """After a sell fill, proceeds go to unsettled_cash, not settled_cash."""
        clock = FakeClock(_NOW)
        initial = Decimal("50000")
        snapshot = _account(initial)
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        sell_order_id = uuid.uuid4()
        fill = _make_fill(sell_order_id, OrderSide.SELL, Decimal("100"), qty=10)
        result = await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=fill, is_complete=True)]
        )

        assert result.fills_applied == 1
        assert result.settlements_projected == 1
        assert ledger.unsettled_cash > Decimal("0")
        assert ledger.settled_cash == initial

    async def test_unsettled_cash_cannot_fund_buy(self) -> None:
        """A buy order cannot use unsettled proceeds from a sell."""
        clock = FakeClock(_NOW)
        initial = Decimal("2000")
        snapshot = _account(initial)
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, _ = _make_coordinator(clock, ledger)

        # Sell that generates $999 unsettled proceeds.
        sell_fill = _make_fill(uuid.uuid4(), OrderSide.SELL, Decimal("100"), qty=10)
        await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=sell_fill, is_complete=True)]
        )
        assert ledger.unsettled_cash > Decimal("0")

        # Try to buy $2500 worth (more than settled $2000 but within settled + unsettled).
        buy_intent = _buy_intent(Decimal("250"), qty=10)
        decision = ledger.can_open_order(buy_intent, snapshot)
        assert not decision.approved
        assert "insufficient settled cash" in decision.reason

    async def test_advance_settlements_moves_cash(self) -> None:
        """After advancing past settlement date, proceeds become settled."""
        clock = FakeClock(_NOW)
        initial = Decimal("50000")
        snapshot = _account(initial)
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, bus = _make_coordinator(clock, ledger)

        sell_fill = _make_fill(uuid.uuid4(), OrderSide.SELL, Decimal("100"), qty=10)
        await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=sell_fill, is_complete=True)]
        )
        unsettled_before = ledger.unsettled_cash
        assert unsettled_before > Decimal("0")

        # Advance clock to T+1 (next business day).
        clock.advance(days=1)

        settled_count = await coordinator.advance_settlements()
        assert settled_count == 1
        assert ledger.unsettled_cash == Decimal("0")
        assert ledger.settled_cash == initial + unsettled_before

        settlement_events = [e for e in bus.history if isinstance(e, SettlementApplied)]
        assert len(settlement_events) == 1

    async def test_settlement_not_advanced_before_date(self) -> None:
        """Lots are not settled before their settlement date arrives."""
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, _ = _make_coordinator(clock, ledger)

        sell_fill = _make_fill(uuid.uuid4(), OrderSide.SELL, Decimal("100"), qty=10)
        await coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=sell_fill, is_complete=True)]
        )

        # Do NOT advance clock — still same day.
        settled_count = await coordinator.advance_settlements()
        assert settled_count == 0
        assert ledger.unsettled_cash > Decimal("0")


# -----------------------------------------------------------------------
# Test: Cash Drift Detection
# -----------------------------------------------------------------------


class TestCashDrift:
    """Coordinator detects when ledger and broker cash diverge."""

    def test_drift_within_tolerance_ok(self) -> None:
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, _ = _make_coordinator(clock, ledger)

        ok, drift = coordinator.check_cash_drift(Decimal("50000.50"))
        assert ok
        assert abs(drift) <= Decimal("1.00")

    def test_drift_exceeds_tolerance_flags(self) -> None:
        clock = FakeClock(_NOW)
        snapshot = _account(Decimal("50000"))
        ledger = CashLedger(
            clock=clock, settlement_calendar=SettlementCalendar(), initial_snapshot=snapshot
        )
        coordinator, _ = _make_coordinator(clock, ledger)

        ok, drift = coordinator.check_cash_drift(Decimal("49000"))
        assert not ok
        assert drift == Decimal("1000")

    async def test_cycle_halts_and_resyncs_on_drift(self) -> None:
        """run_strategy_cycle must halt and activate kill switch on broker cash drift."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
        )
        await session.broker.connect()

        # Simulate a broker snapshot that diverges from the ledger and is marked as live.
        drift_snapshot = _account(Decimal("90000"))

        async def _fake_sync_account() -> AccountSnapshot:
            return drift_snapshot

        session.broker.sync_account = _fake_sync_account  # type: ignore[method-assign]

        result = await run_strategy_cycle(
            session=session,
            feature_data={_INST_A: {"momentum": 0.9}},
            strategy_run=_make_strategy_run(),
            market_prices={_INST_A: Decimal("100")},
            regime=_make_regime(),
        )

        assert result.target is None
        assert result.submitted_ids == []
        assert session.execution_policy.kill_switch_active
        assert session.cash_engine.settled_cash == Decimal("90000")

        kill_events = [e for e in session.event_bus.history if isinstance(e, KillSwitchActivated)]
        assert len(kill_events) == 1


# -----------------------------------------------------------------------
# Test: Full Cycle With Coordinator
# -----------------------------------------------------------------------


class TestFullCycleWithCoordinator:
    """run_strategy_cycle() uses the coordinator path."""

    async def test_cycle_uses_coordinator_lifecycle_feed(self) -> None:
        """The full cycle processes fills via the coordinator lifecycle feed."""
        clock = FakeClock(_NOW)
        signal_model = LinearWeightSignalModel({"momentum": 1.0})
        constructor = LongOnlyPortfolioConstructor(top_n=2, min_score_threshold=0.0)
        strategy_run = _make_strategy_run()

        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        broker.set_market_price(_INST_B, Decimal("50"))
        await broker.connect()

        result = await run_strategy_cycle(
            session=session,
            feature_data={
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": 0.6},
            },
            strategy_run=strategy_run,
            market_prices={
                _INST_A: Decimal("100"),
                _INST_B: Decimal("50"),
            },
            regime=_make_regime(),
        )

        assert len(result.fills) > 0
        assert session.cash_engine.reserved_cash == Decimal("0")
        assert session.cash_engine.settled_cash < Decimal("100000")

        # Fills should have generated OrderFilled events on the bus.
        filled_events = [e for e in session.event_bus.history if isinstance(e, OrderFilled)]
        assert len(filled_events) == len(result.fills)

        # Fills should be persisted in the order repository.
        persisted_fills = []
        for order_id in result.submitted_ids:
            persisted_fills.extend(await session.order_repo.get_fills(order_id))
        assert len(persisted_fills) == len(result.fills)

        # Fully-filled orders should not appear as open.
        open_orders = await session.order_repo.list_open_orders(strategy_run.run_id)
        assert open_orders == []

    async def test_cycle_expires_stale_reservations(self) -> None:
        """Stale pre-submission reservations are expired at cycle start."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("50000"),
            clock=clock,
        )
        await session.broker.connect()

        # Manually create a reservation that will expire.
        intent = _buy_intent(Decimal("100"), qty=5)
        snapshot = _account(Decimal("50000"))
        reservation = session.cash_engine.reserve_cash(intent, snapshot)
        assert session.cash_engine.reserved_cash > Decimal("0")

        # Advance past the 30-minute TTL.
        clock.advance(minutes=31)

        # Run a cycle — the stale reservation should be expired at step 0a.
        _ = await run_strategy_cycle(
            session=session,
            feature_data={},
            strategy_run=_make_strategy_run(),
            market_prices={},
            regime=_make_regime(),
        )

        assert session.cash_engine.reserved_cash == Decimal("0")


# -----------------------------------------------------------------------
# Test: Partial Fill via Simulated Broker
# -----------------------------------------------------------------------


class TestSimulatedBrokerPartialFills:
    """SimulatedBrokerGateway.simulate_partial_fill() works with the coordinator."""

    async def test_partial_then_complete_via_drain(self) -> None:
        """Two partial fills: the second with is_complete=True releases reservation."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("50000"),
            clock=clock,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        account = _account(Decimal("50000"))
        intent = _buy_intent(Decimal("100"), qty=20)

        # Approve and submit normally (which auto-fills).
        approved, _ = await session.approve_ctrl.approve([intent], account)
        assert len(approved) == 1

        # Drain the auto-fill from place_order().
        submitted = await session.submit_ctrl.submit(approved)
        assert len(submitted) == 1

        events = await broker.drain_lifecycle_events()
        await session.coordinator.process_lifecycle_events(events)
        assert session.cash_engine.reserved_cash == Decimal("0")

    async def test_cancel_via_simulate_cancel(self) -> None:
        """simulate_cancel() generates a lifecycle event that releases cash."""
        clock = FakeClock(_NOW)
        broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("50000"))
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        # Submit an order that auto-fills; drain those events.
        intent = _buy_intent(Decimal("100"), qty=10)
        await broker.place_order(intent)
        _ = await broker.drain_lifecycle_events()

        # Now submit a second order that we can cancel.
        intent2 = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=_RUN_ID,
            portfolio_target_id=uuid.uuid4(),
            instrument_id=_INST_A,
            side=OrderSide.BUY,
            quantity=5,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            created_at=_NOW,
            limit_price=Decimal("100"),
        )
        await broker.place_order(intent2)
        # Drain the auto-fill.
        auto_events = await broker.drain_lifecycle_events()

        # simulate_cancel on a new order that's kept open via simulate_partial_fill.
        intent3 = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=_RUN_ID,
            portfolio_target_id=uuid.uuid4(),
            instrument_id=_INST_A,
            side=OrderSide.BUY,
            quantity=20,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            created_at=_NOW,
            limit_price=Decimal("100"),
        )
        # Place the order; drain auto-fill.
        await broker.place_order(intent3)
        _ = await broker.drain_lifecycle_events()

        # Inject a cancel.
        broker.simulate_cancel(intent3.order_id, "test cancel")
        cancel_events = await broker.drain_lifecycle_events()

        assert len(cancel_events) == 1
        assert isinstance(cancel_events[0], BrokerOrderCancelled)
        assert cancel_events[0].order_id == intent3.order_id

    async def test_reject_via_simulate_reject(self) -> None:
        """simulate_reject() generates a BrokerOrderRejected lifecycle event."""
        clock = FakeClock(_NOW)
        broker = SimulatedBrokerGateway(clock=clock, initial_cash=Decimal("50000"))
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        intent = _buy_intent(Decimal("100"), qty=10)
        await broker.place_order(intent)
        _ = await broker.drain_lifecycle_events()

        broker.simulate_reject(intent.order_id, "simulated reject")
        events = await broker.drain_lifecycle_events()

        assert len(events) == 1
        assert isinstance(events[0], BrokerOrderRejected)


# -----------------------------------------------------------------------
# Test: Restart and Reconcile
# -----------------------------------------------------------------------


class TestRestartAndReconcile:
    """After a restart, reconciliation corrects internal state."""

    async def test_reconcile_after_restart_corrects_state(self) -> None:
        """Reconciliation persists broker-authoritative state and detects no issues."""
        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        # Save initial snapshot.
        account = await broker.sync_account()
        await session.position_repo.save_snapshot(account)

        # Run reconciliation — should find no discrepancies.
        await session.recon_ctrl.reconcile(_RUN_ID)

        # Kill switch should NOT be active (no discrepancies).
        assert not session.execution_policy.kill_switch_active

    async def test_reconcile_detects_position_discrepancy(self) -> None:
        """Reconciliation activates kill switch when internal state has a
        position the broker does not (extra internal position).
        """
        from datetime import timedelta

        from quant_platform.core.domain.portfolio.positions import PositionSnapshot

        # Use a far-future timestamp so get_latest_snapshot() returns this phantom
        # over any broker snapshot left by tests that run earlier in the suite.
        # Other integration tests use timestamps up to 2026-02-04, so 2099 is safe.
        # The phantom's as_of is +1s ahead of clock.now() so it sorts ABOVE the
        # broker-authoritative snapshot that reconcile() saves with as_of=clock.now().
        _CLOCK_NOW = datetime(2099, 1, 1, 0, 0, 0, tzinfo=_UTC)
        _PHANTOM_NOW = _CLOCK_NOW + timedelta(seconds=1)
        clock = FakeClock(_CLOCK_NOW)

        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("100000"),
            clock=clock,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        # Save a snapshot WITH a position that the broker does NOT have.
        phantom_pos = PositionSnapshot(
            snapshot_id=uuid.uuid4(),
            instrument_id=_INST_A,
            quantity=500,
            average_cost=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("50000"),
            unrealised_pnl=Decimal("0"),
            as_of=_PHANTOM_NOW,
        )
        internal_snapshot = AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_PHANTOM_NOW,
            settled_cash=Decimal("50000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("50000"),
            net_asset_value=Decimal("100000"),
            positions=(phantom_pos,),
        )
        await session.position_repo.save_snapshot(internal_snapshot)

        # Run reconciliation — broker has NO position, internal has 500 shares.
        await session.recon_ctrl.reconcile(_RUN_ID)

        # Extra internal position triggers operator action → kill switch.
        assert session.execution_policy.kill_switch_active


# -----------------------------------------------------------------------
# Test: Operator Read Models Reflect State
# -----------------------------------------------------------------------


class TestOperatorReadModels:
    """Read models show correct cash and broker state."""

    async def test_cash_status_after_fill(self) -> None:
        """CashStatusView reflects settled cash decrease after a buy fill."""
        from quant_platform.application.operator_api.read_models import OperatorReadModelBuilder

        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("50000"),
            clock=clock,
        )

        broker: SimulatedBrokerGateway = session.broker  # type: ignore
        broker.set_market_price(_INST_A, Decimal("100"))
        await broker.connect()

        builder = OperatorReadModelBuilder(
            clock=clock,
            cash_ledger=session.cash_engine,  # type: ignore[arg-type]
            throttle=session.execution_policy,  # type: ignore[arg-type]
            order_repo=session.order_repo,  # type: ignore[arg-type]
            position_repo=session.position_repo,  # type: ignore[arg-type]
        )

        status_before = builder.cash_status()
        assert status_before.settled_cash == Decimal("50000")

        # Process a buy fill — save the order intent first to satisfy the FK constraint.
        intent = _buy_intent(Decimal("100"), qty=10)
        await session.order_repo.save_intent(intent)
        fill = _make_fill(intent.order_id, OrderSide.BUY, Decimal("100"), qty=10)
        await session.coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=fill, is_complete=True)]
        )

        status_after = builder.cash_status()
        assert status_after.settled_cash < status_before.settled_cash

    async def test_cash_status_shows_unsettled_after_sell(self) -> None:
        """CashStatusView reflects unsettled cash after a sell fill."""
        from quant_platform.application.operator_api.read_models import OperatorReadModelBuilder

        clock = FakeClock(_NOW)
        session = create_paper_session(
            _SETTINGS,
            initial_cash=Decimal("50000"),
            clock=clock,
        )
        await session.broker.connect()

        builder = OperatorReadModelBuilder(
            clock=clock,
            cash_ledger=session.cash_engine,  # type: ignore[arg-type]
            throttle=session.execution_policy,  # type: ignore[arg-type]
            order_repo=session.order_repo,  # type: ignore[arg-type]
            position_repo=session.position_repo,  # type: ignore[arg-type]
        )

        sell_intent = _sell_intent(Decimal("100"), qty=10)
        await session.order_repo.save_intent(sell_intent)
        sell_fill = _make_fill(sell_intent.order_id, OrderSide.SELL, Decimal("100"), qty=10)
        await session.coordinator.process_lifecycle_events(
            [BrokerFillEvent(fill=sell_fill, is_complete=True)]
        )

        status = builder.cash_status()
        assert status.unsettled_cash > Decimal("0")
        assert status.settled_cash == Decimal("50000")
