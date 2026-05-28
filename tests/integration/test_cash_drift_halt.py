"""Chaos test: cash drift exceeds tolerance → kill switch activates.

Scenario: the broker-authoritative account snapshot disagrees with the
local ledger's settled cash by more than ``cash.drift_tolerance_usd``.
The cycle must:

* Resync from broker (``resync_from_broker_snapshot``).
* Activate the execution-policy kill switch.
* Emit ``KillSwitchActivated`` tagged ``strategy_cycle``.
* Return an empty ``CycleResult`` (no orders submitted after halt).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import CashSettings, PlatformSettings, RiskSettings
from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import KillSwitchActivated
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC
_NOW = datetime(2026, 2, 3, 16, 0, 0, tzinfo=_UTC)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="chaos_cash_drift",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=RegimeLabel.RISK_ON,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


@pytest.mark.asyncio
async def test_cash_drift_exceeds_tolerance_halts_cycle() -> None:
    clock = FakeClock(_NOW)
    instrument = uuid.uuid4()
    initial_cash = Decimal("100000")

    settings = PlatformSettings(
        _env_file=None,
        cash=CashSettings(drift_tolerance_usd=Decimal("1.00")),
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.20"),
            max_sector_weight=Decimal("0.50"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.30"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
    )

    session = create_paper_session(
        settings=settings,
        initial_cash=initial_cash,
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )
    session.broker.set_market_price(instrument, Decimal("100"))  # type: ignore[attr-defined]
    await session.broker.connect()

    # Force the account-broker to report a drifted settled cash with
    # source='broker', triggering the drift-halt branch.
    drifted_settled = initial_cash - Decimal("500.00")

    async def _drifted_sync_account() -> AccountSnapshot:
        return AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=clock.now(),
            settled_cash=drifted_settled,
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=drifted_settled,
            net_asset_value=drifted_settled,
            positions=(),
            source="broker",
        )

    session.account_broker.sync_account = _drifted_sync_account  # type: ignore[method-assign]

    purge_calls = 0
    original_purge = session.coordinator.purge_durable_state

    async def _tracked_purge() -> None:
        nonlocal purge_calls
        purge_calls += 1
        await original_purge()

    session.coordinator.purge_durable_state = _tracked_purge  # type: ignore[method-assign]

    result = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.9}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )

    # Cycle halted before any orders were built.
    assert result.target is None
    assert result.signals == []
    assert result.submitted_ids == []

    # Kill switch is active and the event was broadcast.
    assert session.execution_policy.kill_switch_active
    kill_events = [e for e in session.event_bus.history if isinstance(e, KillSwitchActivated)]
    assert kill_events, "KillSwitchActivated must be emitted on cash drift"
    drift_event = next((e for e in kill_events if "cash drift" in e.reason), None)
    assert drift_event is not None
    assert drift_event.activated_by == "cash_drift_guard"
    assert purge_calls == 1


@pytest.mark.asyncio
async def test_strategy_cycle_advances_buy_side_t1_settlement() -> None:
    clock = FakeClock(_NOW)
    instrument = uuid.uuid4()
    initial_cash = Decimal("100000")
    settings = PlatformSettings(
        _env_file=None,
        cash=CashSettings(buy_side_t1_settlement=True),
    )
    session = create_paper_session(
        settings=settings,
        initial_cash=initial_cash,
        clock=clock,
    )
    await session.broker.connect()

    order_id = uuid.uuid4()
    intent = OrderIntent(
        order_id=order_id,
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
        time_in_force=TimeInForce.DAY,
        created_at=clock.now(),
        cash_reservation_id=uuid.uuid4(),
    )
    await session.order_repo.save_intent(intent)
    fill = FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="BRK-BUY-1",
        instrument_id=instrument,
        side=OrderSide.BUY,
        quantity=10,
        fill_price=Decimal("100"),
        commission=Decimal("1"),
        currency="USD",
        executed_at=clock.now(),
        received_at=clock.now(),
    )
    await session.coordinator.process_lifecycle_events(
        [BrokerFillEvent(fill=fill, is_complete=True)]
    )
    assert session.cash_engine.settled_cash == initial_cash
    # T+1 buy debit is tracked internally; sell-proceeds pool stays 0
    assert session.cash_engine.unsettled_cash == Decimal("0")

    clock.set(datetime(2026, 2, 4, 16, 0, 0, tzinfo=_UTC))
    result = await run_strategy_cycle(
        session=session,
        feature_data={},
        strategy_run=_strategy_run(),
        market_prices={},
        regime=_regime(),
    )

    assert result.submitted_ids == []
    assert session.cash_engine.settled_cash == initial_cash - Decimal("1001")
    assert session.cash_engine.unsettled_cash == Decimal("0")
