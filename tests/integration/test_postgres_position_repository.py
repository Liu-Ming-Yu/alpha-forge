"""Integration tests for PostgresPositionRepository against a real Postgres database."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.infrastructure.postgres.repositories import (
    PostgresPositionRepository,
    create_pg_engine,
)

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


def _make_position(instrument_id: uuid.UUID, as_of: datetime) -> PositionSnapshot:
    return PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=100,
        average_cost=Decimal("50.00"),
        market_price=Decimal("55.00"),
        market_value=Decimal("5500.00"),
        unrealised_pnl=Decimal("500.00"),
        as_of=as_of,
        source="broker",
    )


def _make_snapshot(
    *,
    snapshot_id: uuid.UUID | None = None,
    as_of: datetime | None = None,
    positions: tuple[PositionSnapshot, ...] = (),
) -> AccountSnapshot:
    now = as_of or datetime.now(tz=_UTC)
    sid = snapshot_id or uuid.uuid4()
    cash = Decimal("10000.00")
    return AccountSnapshot(
        snapshot_id=sid,
        as_of=now,
        settled_cash=cash,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=cash,
        net_asset_value=cash,
        positions=positions,
        source="broker",
    )


@pytest.mark.asyncio
async def test_save_snapshot_and_get_latest_round_trip() -> None:
    repo = PostgresPositionRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    inst1 = uuid.uuid4()
    inst2 = uuid.uuid4()
    pos1 = _make_position(inst1, now)
    pos2 = _make_position(inst2, now)
    snapshot = _make_snapshot(as_of=now, positions=(pos1, pos2))

    await repo.save_snapshot(snapshot)
    fetched = await repo.get_latest_snapshot()

    assert fetched is not None
    assert fetched.snapshot_id == snapshot.snapshot_id
    assert fetched.settled_cash == snapshot.settled_cash
    fetched_instrument_ids = {p.instrument_id for p in fetched.positions}
    assert inst1 in fetched_instrument_ids
    assert inst2 in fetched_instrument_ids


@pytest.mark.asyncio
async def test_get_snapshot_at_returns_correct_historical() -> None:
    repo = PostgresPositionRepository(create_pg_engine(_postgres_dsn()))
    t1 = datetime.now(tz=_UTC) - timedelta(hours=2)
    t2 = t1 + timedelta(hours=1)

    snap1 = _make_snapshot(as_of=t1)
    snap2 = _make_snapshot(as_of=t2)
    await repo.save_snapshot(snap1)
    await repo.save_snapshot(snap2)

    query_time = t1 + timedelta(minutes=30)
    result = await repo.get_snapshot_at(query_time)
    assert result is not None
    assert result.snapshot_id == snap1.snapshot_id


@pytest.mark.asyncio
async def test_hydrate_loads_position_snapshots() -> None:
    repo = PostgresPositionRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    inst = uuid.uuid4()
    pos = _make_position(inst, now)
    snapshot = _make_snapshot(as_of=now, positions=(pos,))

    await repo.save_snapshot(snapshot)
    fetched = await repo.get_latest_snapshot()

    assert fetched is not None
    assert len(fetched.positions) >= 1
    our_pos = next((p for p in fetched.positions if p.instrument_id == inst), None)
    assert our_pos is not None
    assert isinstance(our_pos.average_cost, Decimal)
    assert isinstance(our_pos.market_price, Decimal)
    assert our_pos.quantity == 100


@pytest.mark.asyncio
async def test_save_snapshot_idempotent_on_duplicate_snapshot_id() -> None:
    repo = PostgresPositionRepository(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)
    sid = uuid.uuid4()
    snapshot = _make_snapshot(snapshot_id=sid, as_of=now)

    await repo.save_snapshot(snapshot)
    await repo.save_snapshot(snapshot)

    from sqlalchemy import text

    engine = create_pg_engine(_postgres_dsn())
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT COUNT(*) AS cnt FROM account_snapshots WHERE snapshot_id = :sid"),
                    {"sid": sid},
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["cnt"] == 1


@pytest.mark.asyncio
async def test_get_snapshot_at_returns_none_when_no_history() -> None:
    repo = PostgresPositionRepository(create_pg_engine(_postgres_dsn()))
    # Query for a time far in the past when no snapshots existed for this test run
    ancient = datetime(2000, 1, 1, tzinfo=_UTC)
    result = await repo.get_snapshot_at(ancient)
    assert result is None
