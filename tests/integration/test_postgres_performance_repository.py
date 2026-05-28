"""Integration tests for PostgresPerformanceRepository against a real Postgres database."""

from __future__ import annotations

import os
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
    SignalGateRecord,
    TextSignalGateRecord,
)
from quant_platform.infrastructure.performance import PostgresPerformanceRepository
from quant_platform.infrastructure.postgres.repositories import create_pg_engine

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


def _nav_snapshot(
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    nav: Decimal = Decimal("100000"),
) -> NavSnapshot:
    return NavSnapshot(
        snapshot_id=uuid.uuid4(),
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        net_asset_value=nav,
        gross_exposure=Decimal("0.80"),
        cash=Decimal("20000"),
        source="runtime",
    )


@pytest.mark.asyncio
async def test_save_and_list_nav_snapshots_ordered() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    srid = uuid.uuid4()
    base = datetime.now(tz=_UTC) - timedelta(hours=3)

    snaps = [_nav_snapshot(srid, base + timedelta(hours=i)) for i in range(3)]
    for s in snaps:
        await repo.save_nav_snapshot(s)

    result = await repo.list_nav_snapshots(srid, limit=10)
    assert len(result) == 3
    for i in range(1, len(result)):
        assert result[i].as_of >= result[i - 1].as_of


@pytest.mark.asyncio
async def test_list_nav_snapshots_respects_limit() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    srid = uuid.uuid4()
    base = datetime.now(tz=_UTC) - timedelta(hours=12)

    for i in range(10):
        await repo.save_nav_snapshot(_nav_snapshot(srid, base + timedelta(hours=i)))

    result = await repo.list_nav_snapshots(srid, limit=5)
    assert len(result) == 5


@pytest.mark.asyncio
async def test_save_nav_snapshot_deduplicates_on_snapshot_id() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    srid = uuid.uuid4()
    snap = _nav_snapshot(srid, datetime.now(tz=_UTC))
    await repo.save_nav_snapshot(snap)
    await repo.save_nav_snapshot(snap)

    from sqlalchemy import text

    engine = create_pg_engine(_postgres_dsn())
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT COUNT(*) AS cnt FROM nav_snapshots WHERE snapshot_id = :sid"),
                    {"sid": snap.snapshot_id},
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["cnt"] == 1


@pytest.mark.asyncio
async def test_record_ic_and_status_gate_passes() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    strategy = f"strategy-{uuid.uuid4()}"
    base = datetime.now(tz=_UTC) - timedelta(days=10)
    ics = [0.08, 0.07, 0.09, 0.06, 0.10]
    for i, ic in enumerate(ics):
        await repo.record_ic(
            TextSignalGateRecord(
                strategy_name=strategy,
                as_of=base + timedelta(days=i),
                daily_ic=ic,
                observations=1,
            )
        )

    status = await repo.status(
        strategy,
        as_of=base + timedelta(days=10),
        min_observations=5,
        min_ic=0.05,
        max_negative_streak=3,
    )
    assert status.passed is True
    assert status.observations >= 5


@pytest.mark.asyncio
async def test_record_ic_status_gate_fails_on_low_ic() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    strategy = f"strategy-low-{uuid.uuid4()}"
    base = datetime.now(tz=_UTC) - timedelta(days=5)
    for i in range(5):
        await repo.record_ic(
            TextSignalGateRecord(
                strategy_name=strategy,
                as_of=base + timedelta(days=i),
                daily_ic=0.01,
                observations=1,
            )
        )

    status = await repo.status(
        strategy,
        as_of=base + timedelta(days=5),
        min_observations=5,
        min_ic=0.05,
    )
    assert status.passed is False


@pytest.mark.asyncio
async def test_signal_status_drawdown_gate() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)

    name_pass = f"sig-pass-{uuid.uuid4()}"
    await repo.record_signal_observation(
        SignalGateRecord(
            signal_name=name_pass,
            signal_type="test",
            as_of=now,
            daily_ic=0.08,
            observations=25,
            drawdown=-0.05,
            turnover=0.40,
        )
    )
    status_pass = await repo.signal_status(
        name_pass,
        "test",
        as_of=now + timedelta(seconds=1),
        min_observations=1,
        min_ic=0.05,
        drawdown_limit=-0.10,
        turnover_limit=0.50,
    )
    assert status_pass.passed is True

    name_fail = f"sig-fail-{uuid.uuid4()}"
    await repo.record_signal_observation(
        SignalGateRecord(
            signal_name=name_fail,
            signal_type="test",
            as_of=now,
            daily_ic=0.08,
            observations=25,
            drawdown=-0.15,
            turnover=0.40,
        )
    )
    status_fail = await repo.signal_status(
        name_fail,
        "test",
        as_of=now + timedelta(seconds=1),
        min_observations=1,
        min_ic=0.05,
        drawdown_limit=-0.10,
        turnover_limit=0.50,
    )
    assert status_fail.passed is False


@pytest.mark.asyncio
async def test_save_runtime_heartbeat_and_latest_broker_health() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)

    observation = BrokerHealthObservation(
        observed_at=now,
        status="connected",
        latency_ms=12.5,
        last_heartbeat_at=now,
        detail="integration test",
    )
    await repo.save_broker_health(observation)
    latest = await repo.latest_broker_health()
    assert latest is not None
    assert latest.status == "connected"
    assert latest.latency_ms == 12.5


@pytest.mark.asyncio
async def test_save_broker_smoke_and_latest() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)

    observation = BrokerSmokeObservation(
        observed_at=now,
        status="connected",
        host="127.0.0.1",
        port=4002,
        client_id=42,
        latency_ms=8.0,
        account_status="ok",
        positions_status="ok",
        open_orders_status="ok",
        detail="integration test",
    )
    await repo.save_broker_smoke(observation)
    latest = await repo.latest_broker_smoke()
    assert latest is not None
    assert latest.passed is True
    assert latest.host == "127.0.0.1"


@pytest.mark.asyncio
async def test_save_paper_lifecycle_and_latest() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)

    observation = PaperLifecycleObservation(
        observed_at=now,
        status="passed",
        host="127.0.0.1",
        port=4002,
        client_id=99,
        instrument_id=uuid.uuid4(),
        broker_order_id="TEST-PAPER-001",
        max_notional_usd=Decimal("100"),
        limit_price=Decimal("50.00"),
        quantity=1,
        ack_status="ok",
        cancel_status="ok",
        stale_open_order_count=0,
        detail="integration test",
    )
    await repo.save_paper_lifecycle(observation)
    latest = await repo.latest_paper_lifecycle()
    assert latest is not None
    assert latest.passed is True
    assert latest.broker_order_id == "TEST-PAPER-001"


@pytest.mark.asyncio
async def test_runtime_heartbeat_upserts_by_component() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    component = f"test-engine-{uuid.uuid4()}"

    await repo.save_runtime_heartbeat(
        RuntimeHeartbeat(component=component, as_of=now, status="ok", detail="first")
    )
    await repo.save_runtime_heartbeat(
        RuntimeHeartbeat(
            component=component,
            as_of=now + timedelta(seconds=1),
            status="degraded",
            detail="second",
        )
    )

    latest = await repo.latest_runtime_heartbeat(component)
    assert latest is not None
    assert latest.status == "degraded"
    assert latest.detail == "second"
