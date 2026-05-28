"""Durable-backend integration tests for Postgres-backed adapters."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    PaperLifecycleObservation,
    RuntimeHeartbeat,
    SignalGateRecord,
)
from quant_platform.core.domain.research import FeatureVector
from quant_platform.core.events import OrderApproved, OrderRejected
from quant_platform.infrastructure.event_bus import RedisStreamsEventBus
from quant_platform.infrastructure.performance import PostgresPerformanceRepository
from quant_platform.infrastructure.postgres.feature_repository import (
    PostgresFeatureRepository,
)
from quant_platform.infrastructure.postgres.model_registry import PostgresModelRegistry
from quant_platform.infrastructure.postgres.repositories import (
    PostgresAuditSink,
    PostgresOrderRepository,
    create_pg_engine,
)

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for durable Redis tests")
    return url


def _feature_vector(
    *,
    instrument_id: uuid.UUID,
    feature_set_version: str,
    as_of: datetime,
    available_at: datetime | None = None,
) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=as_of,
        feature_set_version=feature_set_version,
        features={"momentum_1m": 0.25},
        strategy_run_id=uuid.uuid4(),
        artifact_uri="file:///tmp/durable-test",
        available_at=available_at or as_of,
    )


@pytest.mark.asyncio
async def test_postgres_audit_listing_uses_migration_schema() -> None:
    engine = create_pg_engine(_postgres_dsn())
    sink = PostgresAuditSink(engine)
    now = datetime.now(tz=_UTC)
    order_id = uuid.uuid4()

    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=now,
            order_id=order_id,
            reason="durable audit test",
        ),
        context={"test": "durable_backends"},
    )

    rows = await sink.list_events(event_type="OrderRejected", limit=50)

    assert rows
    row = next(row for row in rows if row["event_payload"]["order_id"] == str(order_id))
    assert row["audit_id"]
    assert row["event_type"] == "OrderRejected"
    assert row["event_payload"]["order_id"] == str(order_id)
    assert row["context"]["test"] == "durable_backends"


@pytest.mark.asyncio
async def test_postgres_feature_repository_rejects_duplicate_natural_key() -> None:
    repo = PostgresFeatureRepository(create_pg_engine(_postgres_dsn()))
    instrument_id = uuid.uuid4()
    version = f"durable-{uuid.uuid4()}"
    as_of = datetime.now(tz=_UTC)

    await repo.store_vector(
        _feature_vector(
            instrument_id=instrument_id,
            feature_set_version=version,
            as_of=as_of,
        )
    )

    with pytest.raises(ValueError, match="Duplicate FeatureVector"):
        await repo.store_vector(
            _feature_vector(
                instrument_id=instrument_id,
                feature_set_version=version,
                as_of=as_of,
            )
        )


@pytest.mark.asyncio
async def test_postgres_feature_lookup_and_prune() -> None:
    repo = PostgresFeatureRepository(create_pg_engine(_postgres_dsn()))
    instrument_id = uuid.uuid4()
    version = f"durable-{uuid.uuid4()}"
    now = datetime.now(tz=_UTC)
    stale_as_of = now - timedelta(days=10)
    fresh_as_of = now

    stale = _feature_vector(
        instrument_id=instrument_id,
        feature_set_version=version,
        as_of=stale_as_of,
    )
    fresh = _feature_vector(
        instrument_id=instrument_id,
        feature_set_version=version,
        as_of=fresh_as_of,
    )
    await repo.store_vector(stale)
    await repo.store_vector(fresh)

    latest = await repo.get_vectors([instrument_id], version, now + timedelta(seconds=1))
    assert [v.vector_id for v in latest] == [fresh.vector_id]

    deleted = await repo.prune(now - timedelta(days=5))
    assert deleted >= 1

    remaining = await repo.get_vectors([instrument_id], version, now + timedelta(seconds=1))
    assert [v.vector_id for v in remaining] == [fresh.vector_id]


@pytest.mark.asyncio
async def test_postgres_feature_repository_excludes_late_available_at() -> None:
    repo = PostgresFeatureRepository(create_pg_engine(_postgres_dsn()))
    instrument_id = uuid.uuid4()
    version = f"durable-available-at-{uuid.uuid4()}"
    decision_time = datetime.now(tz=_UTC)
    late = _feature_vector(
        instrument_id=instrument_id,
        feature_set_version=version,
        as_of=decision_time,
        available_at=decision_time + timedelta(minutes=1),
    )
    visible = _feature_vector(
        instrument_id=instrument_id,
        feature_set_version=version,
        as_of=decision_time - timedelta(minutes=1),
        available_at=decision_time,
    )

    await repo.store_vector(late)
    await repo.store_vector(visible)

    result = await repo.get_vectors([instrument_id], version, decision_time)

    assert [vector.vector_id for vector in result] == [visible.vector_id]


@pytest.mark.asyncio
async def test_postgres_order_repository_allows_partial_fills_and_dedupes_execution_id() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    order_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    broker_order_id = f"durable-partial-{uuid.uuid4()}"
    intent = OrderIntent(
        order_id=order_id,
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=now,
        limit_price=Decimal("100"),
        cash_reservation_id=uuid.uuid4(),
    )
    await repo.save_intent(intent)

    first = FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id=broker_order_id,
        broker_execution_id="exec-1",
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=40,
        fill_price=Decimal("100"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=now,
        received_at=now,
    )
    second = FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id=broker_order_id,
        broker_execution_id="exec-2",
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=60,
        fill_price=Decimal("100"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=now,
        received_at=now,
    )
    duplicate_execution = FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id=broker_order_id,
        broker_execution_id="exec-2",
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=60,
        fill_price=Decimal("100"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=now,
        received_at=now,
    )

    await repo.save_fill(first)
    await repo.save_fill(second)
    await repo.save_fill(duplicate_execution)

    fills = await repo.get_fills(order_id)
    assert [fill.broker_execution_id for fill in fills] == ["exec-1", "exec-2"]


@pytest.mark.asyncio
async def test_postgres_model_registry_feature_job_schedule() -> None:
    registry = PostgresModelRegistry(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    strategy_name = f"durable-{uuid.uuid4()}"
    model = await registry.register_model(
        strategy_name=strategy_name,
        model_version="0.1.0",
        feature_set_version="1.0.0",
        as_of=now,
        metadata={"test": "durable_backends"},
    )
    job = await registry.schedule_feature_job(
        model_id=model.model_id,
        strategy_name=strategy_name,
        feature_set_version=model.feature_set_version,
        interval_seconds=60.0,
        as_of=now,
    )

    due = await registry.due_feature_jobs(now)
    assert job.job_id in {j.job_id for j in due}

    await registry.mark_job_completed(job.job_id, run_at=now, success=True)
    due_soon = await registry.due_feature_jobs(now + timedelta(seconds=30))
    assert job.job_id not in {j.job_id for j in due_soon}

    due_later = await registry.due_feature_jobs(now + timedelta(seconds=61))
    assert job.job_id in {j.job_id for j in due_later}


@pytest.mark.asyncio
async def test_postgres_performance_repository_operational_readiness_state() -> None:
    repo = PostgresPerformanceRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    signal_name = f"xgb-{uuid.uuid4()}"
    await repo.record_signal_observation(
        SignalGateRecord(
            signal_name=signal_name,
            signal_type="xgboost",
            as_of=now,
            daily_ic=0.08,
            drawdown=-0.02,
            turnover=0.30,
        )
    )
    await repo.save_runtime_heartbeat(
        RuntimeHeartbeat(
            component=signal_name,
            as_of=now,
            status="ok",
            detail="durable test",
        )
    )
    await repo.save_broker_health(
        BrokerHealthObservation(
            observed_at=now,
            status="connected",
            latency_ms=8.0,
            last_heartbeat_at=now,
            detail="durable test",
        )
    )
    instrument_id = uuid.uuid4()
    await repo.save_broker_smoke(
        BrokerSmokeObservation(
            observed_at=now,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=990,
            latency_ms=8.0,
            account_status="ok",
            positions_status="ok",
            open_orders_status="ok",
            detail="durable test",
        )
    )
    await repo.save_paper_lifecycle(
        PaperLifecycleObservation(
            observed_at=now,
            status="passed",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            instrument_id=instrument_id,
            broker_order_id="100",
            max_notional_usd=Decimal("100"),
            limit_price=Decimal("50"),
            quantity=1,
            ack_status="ok",
            cancel_status="ok",
            stale_open_order_count=0,
            detail="durable test",
        )
    )

    status = await repo.signal_status(
        signal_name,
        "xgboost",
        as_of=now + timedelta(seconds=1),
        min_observations=1,
        min_ic=0.05,
        drawdown_limit=-0.10,
        turnover_limit=0.50,
    )
    heartbeat = await repo.latest_runtime_heartbeat(signal_name)
    broker = await repo.latest_broker_health()
    smoke = await repo.latest_broker_smoke()
    lifecycle = await repo.latest_paper_lifecycle()

    assert status.passed
    assert heartbeat is not None
    assert heartbeat.status == "ok"
    assert broker is not None
    assert broker.status == "connected"
    assert smoke is not None
    assert smoke.passed
    assert lifecycle is not None
    assert lifecycle.passed


@pytest.mark.asyncio
async def test_redis_streams_redelivers_failed_consumer_before_ack() -> None:
    prefix = f"qp:test:{uuid.uuid4()}"
    bus = RedisStreamsEventBus(
        redis_url=_redis_url(),
        stream_prefix=prefix,
        block_ms=50,
        use_consumer_groups=True,
    )
    now = datetime.now(tz=_UTC)
    event = OrderApproved(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=uuid.uuid4(),
        reservation_id=uuid.uuid4(),
    )
    await bus.publish(event)

    consumer = bus.subscribe(OrderApproved, "durable-redelivery")
    first = await asyncio.wait_for(anext(consumer), timeout=2)
    assert first.event_id == event.event_id
    with pytest.raises(RuntimeError, match="handler failed"):
        await consumer.athrow(RuntimeError("handler failed"))

    retry_consumer = bus.subscribe(OrderApproved, "durable-redelivery")
    retried = await asyncio.wait_for(anext(retry_consumer), timeout=2)
    assert retried.event_id == event.event_id

    next_read = asyncio.create_task(anext(retry_consumer))
    await asyncio.sleep(0.1)
    next_read.cancel()
    with suppress(asyncio.CancelledError):
        await next_read
    await retry_consumer.aclose()
