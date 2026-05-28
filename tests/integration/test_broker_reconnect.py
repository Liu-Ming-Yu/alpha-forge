"""Chaos test: broker disconnects and reconnects across strategy cycles.

Scenario:
* Cycle 1 runs end-to-end, submits an order.
* The broker is forcibly disconnected.
* Cycle 2 runs; ``BrokerSessionSupervisor`` must reconnect before the
  submit phase so that no orders are dropped, *and* no order is submitted
  twice (idempotency contract).

The simulated broker dedupes by ``order_id``; this test additionally
asserts that the session emits BrokerSessionHealthChanged events for
both states and that each outbound order has a unique order_id so no
duplicate submission can occur.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import BrokerSettings, PlatformSettings, RiskSettings
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import BrokerSessionHealthChanged, OrderSubmitted
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC
_NOW = datetime(2026, 2, 3, 13, 30, 0, tzinfo=_UTC)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="chaos_broker_reconnect",
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
async def test_broker_reconnect_mid_flow_does_not_duplicate_orders() -> None:
    clock = FakeClock(_NOW)
    instrument = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(
            heartbeat_interval_seconds=0.0,
            max_consecutive_health_failures=1,
            reconnect_base_delay=0.0,
            reconnect_max_delay=0.0,
        ),
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
        initial_cash=Decimal("200000"),
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )
    session.broker.set_market_price(instrument, Decimal("100"))  # type: ignore[attr-defined]
    await session.broker.connect()

    result1 = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.9}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )
    first_submitted_ids = set(result1.submitted_ids)
    assert first_submitted_ids

    # Chaos event: broker session drops between cycles.
    await session.broker.disconnect()

    result2 = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.85}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )

    second_submitted_ids = set(result2.submitted_ids)

    # Cycle must have recovered (broker reconnected) — it may or may not
    # submit new orders depending on target delta; the critical invariant
    # is that no order_id from cycle 1 is reused in cycle 2.
    assert first_submitted_ids.isdisjoint(second_submitted_ids), (
        "Reconnect must not cause an order_id to be resubmitted"
    )

    # Health events prove a disconnect+reconnect cycle occurred.
    health_events = [
        e for e in session.event_bus.history if isinstance(e, BrokerSessionHealthChanged)
    ]
    statuses = {e.current_status for e in health_events}
    assert "disconnected" in statuses
    assert "connected" in statuses

    # Every OrderSubmitted event must reference a unique order_id.
    submitted_events = [e for e in session.event_bus.history if isinstance(e, OrderSubmitted)]
    order_ids = [e.order_id for e in submitted_events]
    assert len(order_ids) == len(set(order_ids)), (
        "Duplicate order_ids in OrderSubmitted history violates idempotency"
    )

    # Kill switch should NOT fire for a routine reconnect.
    assert not session.execution_policy.kill_switch_active
