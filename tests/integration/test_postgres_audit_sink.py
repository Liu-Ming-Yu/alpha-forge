"""Integration tests for PostgresAuditSink against a real Postgres database."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from quant_platform.core.events import OrderApproved, OrderRejected
from quant_platform.infrastructure.postgres.repositories import (
    PostgresAuditSink,
    create_pg_engine,
)

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


async def _postgres_clock(engine) -> datetime:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT clock_timestamp()"))
        return result.scalar_one()


@pytest.mark.asyncio
async def test_record_and_list_events_round_trip() -> None:
    sink = PostgresAuditSink(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    order_id = uuid.uuid4()
    event = OrderRejected(
        event_id=uuid.uuid4(),
        occurred_at=now,
        order_id=order_id,
        reason="audit_sink_round_trip_test",
    )
    ctx = {"source": "test_postgres_audit_sink", "run_id": str(uuid.uuid4())}

    await sink.record(event, context=ctx)

    rows = await sink.list_events(event_type="OrderRejected", limit=50)
    assert rows
    our_row = next((r for r in rows if r["event_payload"].get("order_id") == str(order_id)), None)
    assert our_row is not None
    assert our_row["event_type"] == "OrderRejected"
    assert our_row["event_payload"]["reason"] == "audit_sink_round_trip_test"
    assert our_row["context"]["source"] == "test_postgres_audit_sink"
    assert our_row["audit_id"]


@pytest.mark.asyncio
async def test_list_events_filter_by_event_type() -> None:
    sink = PostgresAuditSink(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    tag = str(uuid.uuid4())

    approved_id = uuid.uuid4()
    rejected_id = uuid.uuid4()

    await sink.record(
        OrderApproved(
            event_id=uuid.uuid4(),
            occurred_at=now,
            order_id=approved_id,
            reservation_id=uuid.uuid4(),
        ),
        context={"tag": tag},
    )
    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=now,
            order_id=rejected_id,
            reason="filter_test",
        ),
        context={"tag": tag},
    )

    approved_rows = await sink.list_events(event_type="OrderApproved", limit=100)
    approved_ids = {r["event_payload"]["order_id"] for r in approved_rows}
    assert str(approved_id) in approved_ids
    assert str(rejected_id) not in approved_ids


@pytest.mark.asyncio
async def test_list_events_filter_by_since_and_until() -> None:
    # audit_log.recorded_at is set by the DB to now() at INSERT transaction start.
    # To guarantee strict ordering e1.recorded_at < t_since < e2.recorded_at <
    # t_until < e3.recorded_at, we must place each boundary timestamp AFTER a real
    # sleep gap (so DB clock is clearly past it) AND BEFORE the next insert.
    # Using 1.0s gaps gives plenty of margin over Windows timer resolution and minor
    # host/DB clock skew.
    # Pattern: insert e1, sleep 1s, sample t_since, sleep 1s, insert e2,
    #          sleep 1s, sample t_until, sleep 1s, insert e3.
    engine = create_pg_engine(_postgres_dsn())
    sink = PostgresAuditSink(engine)
    id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=datetime.now(tz=_UTC),
            order_id=id1,
            reason="time_filter_test",
        ),
        context={},
    )
    await asyncio.sleep(1.0)
    t_since = await _postgres_clock(engine)
    await asyncio.sleep(1.0)

    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=datetime.now(tz=_UTC),
            order_id=id2,
            reason="time_filter_test",
        ),
        context={},
    )
    await asyncio.sleep(1.0)
    t_until = await _postgres_clock(engine)
    await asyncio.sleep(1.0)

    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=datetime.now(tz=_UTC),
            order_id=id3,
            reason="time_filter_test",
        ),
        context={},
    )

    rows = await sink.list_events(
        event_type="OrderRejected",
        since=t_since,
        until=t_until,
        limit=100,
    )
    ids_in_window = {r["event_payload"]["order_id"] for r in rows}
    assert str(id2) in ids_in_window
    assert str(id1) not in ids_in_window
    assert str(id3) not in ids_in_window


@pytest.mark.asyncio
async def test_list_events_offset_pagination() -> None:
    sink = PostgresAuditSink(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    run_tag = str(uuid.uuid4())
    inserted_ids = []
    for _ in range(5):
        oid = uuid.uuid4()
        inserted_ids.append(oid)
        await sink.record(
            OrderRejected(
                event_id=uuid.uuid4(),
                occurred_at=now,
                order_id=oid,
                reason=f"pagination_test_{run_tag}",
            ),
            context={"run_tag": run_tag},
        )

    all_rows = await sink.list_events(event_type="OrderRejected", limit=1000)
    our_rows = [r for r in all_rows if r["event_payload"].get("reason", "").endswith(run_tag)]

    page1 = await sink.list_events(event_type="OrderRejected", limit=2, offset=0)
    our_p1 = [r for r in page1 if r["event_payload"].get("reason", "").endswith(run_tag)]

    page2 = await sink.list_events(event_type="OrderRejected", limit=2, offset=2)
    our_p2 = [r for r in page2 if r["event_payload"].get("reason", "").endswith(run_tag)]

    # We have 5 rows; offset 2 with limit 2 should give different rows than offset 0
    all_audit_ids = {r["audit_id"] for r in our_p1}
    page2_audit_ids = {r["audit_id"] for r in our_p2}
    assert all_audit_ids.isdisjoint(page2_audit_ids) or len(our_rows) < 4


@pytest.mark.asyncio
async def test_list_events_no_filter_returns_results() -> None:
    sink = PostgresAuditSink(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    await sink.record(
        OrderRejected(
            event_id=uuid.uuid4(),
            occurred_at=now,
            order_id=uuid.uuid4(),
            reason="no_filter_test",
        ),
        context={},
    )
    rows = await sink.list_events(limit=10)
    assert len(rows) >= 1
