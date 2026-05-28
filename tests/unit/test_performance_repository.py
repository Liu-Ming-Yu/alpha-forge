from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    NavSnapshot,
    PaperLifecycleObservation,
    RuntimeHeartbeat,
    ShadowPaperParityRecord,
    SignalGateRecord,
    TextSignalGateRecord,
)
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository

_NOW = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_performance_repository_reports_sharpe_and_drawdown() -> None:
    repo = InMemoryPerformanceRepository()
    run_id = uuid.uuid4()
    navs = [Decimal("100"), Decimal("110"), Decimal("105"), Decimal("120")]
    for idx, nav in enumerate(navs):
        await repo.save_nav_snapshot(
            NavSnapshot(
                snapshot_id=uuid.uuid4(),
                strategy_run_id=run_id,
                as_of=_NOW + timedelta(days=idx),
                net_asset_value=nav,
                gross_exposure=Decimal(str(idx)),
                cash=Decimal("10"),
            )
        )

    report = await repo.performance_report(run_id, as_of=_NOW + timedelta(days=4), window=90)

    assert report.observations == 4
    assert report.rolling_sharpe != 0
    assert report.max_drawdown < 0
    assert report.gross_turnover == 3.0


@pytest.mark.asyncio
async def test_text_gate_status_tracks_ic_and_negative_streak() -> None:
    repo = InMemoryPerformanceRepository()
    for idx, ic in enumerate([0.10, 0.06, -0.02, -0.01]):
        await repo.record_ic(
            TextSignalGateRecord(
                strategy_name="xsec",
                as_of=_NOW + timedelta(days=idx),
                daily_ic=ic,
            )
        )

    status = await repo.status(
        "xsec",
        as_of=_NOW + timedelta(days=4),
        min_observations=4,
        min_ic=0.02,
        max_negative_streak=3,
    )

    assert status.observations == 4
    assert status.negative_streak == 2
    assert status.passed

    signal_status = await repo.signal_status(
        "xsec",
        "text",
        as_of=_NOW + timedelta(days=4),
        min_observations=4,
        min_ic=0.02,
        max_negative_streak=3,
    )
    assert signal_status.passed


@pytest.mark.asyncio
async def test_signal_gate_tracks_drawdown_turnover_and_state() -> None:
    repo = InMemoryPerformanceRepository()
    for idx, record in enumerate(
        [
            SignalGateRecord(
                signal_name="xgb-shadow",
                signal_type="xgboost",
                as_of=_NOW + timedelta(days=idx),
                daily_ic=0.08,
                drawdown=-0.03,
                turnover=0.40,
            )
            for idx in range(3)
        ]
    ):
        await repo.record_signal_observation(record)

    status = await repo.signal_status(
        "xgb-shadow",
        "xgboost",
        as_of=_NOW + timedelta(days=3),
        min_observations=3,
        min_ic=0.05,
        max_negative_streak=3,
        drawdown_limit=-0.10,
        turnover_limit=0.50,
    )

    assert status.passed
    assert status.state.value == "ready"
    assert status.max_drawdown == -0.03
    assert status.max_turnover == 0.40


@pytest.mark.asyncio
async def test_shadow_paper_parity_requires_20_clean_trading_days() -> None:
    repo = InMemoryPerformanceRepository()
    for idx in range(20):
        await repo.save_shadow_paper_parity(
            ShadowPaperParityRecord(
                parity_id=uuid.uuid4(),
                signal_name="text",
                signal_type="text",
                trading_day=(_NOW + timedelta(days=idx)).date(),
                as_of=_NOW + timedelta(days=idx),
                instruments_compared=50,
                missing_instruments=0,
                max_target_weight_diff_bps=0.5,
                order_side_mismatches=0,
            )
        )

    status = await repo.shadow_paper_parity_status(
        "text",
        "text",
        as_of=_NOW + timedelta(days=20),
        min_trading_days=20,
        max_target_weight_diff_bps=1.0,
    )

    assert status.passed
    assert status.trading_days == 20
    assert status.blockers == ()

    await repo.save_shadow_paper_parity(
        ShadowPaperParityRecord(
            parity_id=uuid.uuid4(),
            signal_name="text",
            signal_type="text",
            trading_day=(_NOW + timedelta(days=21)).date(),
            as_of=_NOW + timedelta(days=21),
            instruments_compared=50,
            missing_instruments=1,
            max_target_weight_diff_bps=0.5,
            order_side_mismatches=0,
        )
    )
    failed = await repo.shadow_paper_parity_status(
        "text",
        "text",
        as_of=_NOW + timedelta(days=21),
        min_trading_days=20,
        max_target_weight_diff_bps=1.0,
    )

    assert not failed.passed
    assert failed.missing_instruments == 1


@pytest.mark.asyncio
async def test_operational_readiness_evidence_round_trips() -> None:
    repo = InMemoryPerformanceRepository()
    heartbeat = RuntimeHeartbeat(
        component="supervisor",
        as_of=_NOW,
        status="ok",
        detail="cycle complete",
    )
    broker = BrokerHealthObservation(
        observed_at=_NOW,
        status="connected",
        latency_ms=12.5,
        last_heartbeat_at=_NOW,
        detail="paper gateway",
    )

    await repo.save_runtime_heartbeat(heartbeat)
    await repo.save_broker_health(broker)
    smoke = BrokerSmokeObservation(
        observed_at=_NOW,
        status="connected",
        host="host.docker.internal",
        port=4002,
        client_id=990,
        latency_ms=12.5,
        account_status="ok",
        positions_status="ok",
        open_orders_status="ok",
        detail="read-only smoke",
    )
    lifecycle = PaperLifecycleObservation(
        observed_at=_NOW,
        status="passed",
        host="host.docker.internal",
        port=4002,
        client_id=991,
        instrument_id=uuid.uuid4(),
        broker_order_id="100",
        max_notional_usd=Decimal("100"),
        limit_price=Decimal("50"),
        quantity=1,
        ack_status="ok",
        cancel_status="ok",
        stale_open_order_count=0,
        detail="paper lifecycle",
    )
    await repo.save_broker_smoke(smoke)
    await repo.save_paper_lifecycle(lifecycle)

    assert await repo.latest_runtime_heartbeat("supervisor") == heartbeat
    assert await repo.latest_broker_health() == broker
    assert await repo.latest_broker_smoke() == smoke
    assert await repo.latest_paper_lifecycle() == lifecycle
