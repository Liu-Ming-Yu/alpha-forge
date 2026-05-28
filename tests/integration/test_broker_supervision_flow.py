"""Integration tests for broker supervision in strategy cycles."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import BrokerSettings, PlatformSettings, RiskSettings
from quant_platform.core.contracts import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import BrokerSessionHealthChanged, KillSwitchActivated
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="broker_supervision_test",
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
async def test_cycle_recovers_after_disconnect_and_emits_health_events() -> None:
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
        initial_cash=Decimal("100000"),
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )

    # First cycle: healthy path.
    result1 = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.9}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )
    assert len(result1.submitted_ids) > 0

    # Force disconnect before the next cycle; supervisor should reconnect.
    await session.broker.disconnect()

    result2 = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.8}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )
    assert result2.target is not None
    assert not session.execution_policy.kill_switch_active

    health_events = [
        e for e in session.event_bus.history if isinstance(e, BrokerSessionHealthChanged)
    ]
    assert len(health_events) >= 2
    statuses = [e.current_status for e in health_events]
    assert "disconnected" in statuses
    assert "connected" in statuses


@pytest.mark.asyncio
async def test_unrecoverable_broker_health_triggers_paper_gate_blocker() -> None:
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
    )
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("100000"),
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )

    async def _fail_connect() -> None:
        raise RuntimeError("broker unavailable")

    async def _bad_health() -> BrokerHealth:
        return BrokerHealth(
            status=BrokerHealthStatus.DISCONNECTED,
            latency_ms=0,
            last_heartbeat_at=clock.now(),
            detail="forced test failure",
        )

    session.broker.connect = _fail_connect  # type: ignore[method-assign]
    session.broker.health_check = _bad_health  # type: ignore[method-assign]
    session.account_broker = session.broker

    result = await run_strategy_cycle(
        session=session,
        feature_data={instrument: {"momentum": 0.9}},
        strategy_run=_strategy_run(),
        market_prices={instrument: Decimal("100")},
        regime=_regime(),
    )
    assert result.target is None
    assert result.submitted_ids == []
    assert session.execution_policy.kill_switch_active
    kill_events = [e for e in session.event_bus.history if isinstance(e, KillSwitchActivated)]
    assert len(kill_events) >= 1
