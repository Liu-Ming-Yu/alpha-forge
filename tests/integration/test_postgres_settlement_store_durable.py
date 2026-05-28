"""Integration tests for PostgresPendingSettlementStore and
PostgresCompletedOrderHintStore against a real Postgres database."""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.settlement import SettlementLot, SettlementStatus
from quant_platform.infrastructure.postgres.repositories import create_pg_engine
from quant_platform.services.execution_service.stores.pending_settlement_store import (
    PostgresCompletedOrderHintStore,
    PostgresPendingSettlementStore,
)

pytestmark = pytest.mark.integration_durable


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


def _make_lot(*, net_proceeds: Decimal = Decimal("500.00")) -> SettlementLot:
    trade_date = date.today()
    return SettlementLot(
        lot_id=uuid.uuid4(),
        fill_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        trade_date=trade_date,
        settlement_date=trade_date + timedelta(days=2),
        gross_proceeds=net_proceeds + Decimal("1.00"),
        commission=Decimal("1.00"),
        net_proceeds=net_proceeds,
        currency="USD",
        status=SettlementStatus.PENDING,
    )


@pytest.mark.asyncio
async def test_pending_settlement_upsert_and_list() -> None:
    store = PostgresPendingSettlementStore(create_pg_engine(_postgres_dsn()))
    run_id = uuid.uuid4()
    order_id = uuid.uuid4()
    lot = _make_lot()

    await store.upsert(lot, run_id=run_id, order_id=order_id)
    lots = await store.list_all(run_id=run_id)

    assert any(candidate.lot_id == lot.lot_id for candidate in lots)
    our_lot = next(candidate for candidate in lots if candidate.lot_id == lot.lot_id)
    assert our_lot.net_proceeds == lot.net_proceeds
    assert our_lot.currency == "USD"


@pytest.mark.asyncio
async def test_pending_settlement_delete() -> None:
    store = PostgresPendingSettlementStore(create_pg_engine(_postgres_dsn()))
    run_id = uuid.uuid4()
    lot = _make_lot()

    await store.upsert(lot, run_id=run_id, order_id=uuid.uuid4())
    await store.delete(lot.lot_id)

    lots = await store.list_all(run_id=run_id)
    assert not any(candidate.lot_id == lot.lot_id for candidate in lots)


@pytest.mark.asyncio
async def test_pending_settlement_upsert_updates_on_conflict() -> None:
    store = PostgresPendingSettlementStore(create_pg_engine(_postgres_dsn()))
    run_id = uuid.uuid4()
    order_id = uuid.uuid4()
    lot = _make_lot(net_proceeds=Decimal("300.00"))
    await store.upsert(lot, run_id=run_id, order_id=order_id)

    updated_lot = SettlementLot(
        lot_id=lot.lot_id,
        fill_id=lot.fill_id,
        instrument_id=lot.instrument_id,
        trade_date=lot.trade_date,
        settlement_date=lot.settlement_date,
        gross_proceeds=Decimal("801.00"),
        commission=Decimal("1.00"),
        net_proceeds=Decimal("800.00"),
        currency="USD",
        status=SettlementStatus.PENDING,
    )
    await store.upsert(updated_lot, run_id=run_id, order_id=order_id)

    lots = await store.list_all(run_id=run_id)
    our_lots = [candidate for candidate in lots if candidate.lot_id == lot.lot_id]
    assert len(our_lots) == 1
    assert our_lots[0].net_proceeds == Decimal("800.00")


@pytest.mark.asyncio
async def test_completed_order_hint_add_list_remove() -> None:
    store = PostgresCompletedOrderHintStore(create_pg_engine(_postgres_dsn()))
    run_id = uuid.uuid4()
    order_id = uuid.uuid4()

    await store.add(order_id, run_id=run_id)
    all_ids = await store.list_all(run_id=run_id)
    assert order_id in all_ids

    await store.remove(order_id)
    remaining = await store.list_all(run_id=run_id)
    assert order_id not in remaining


@pytest.mark.asyncio
async def test_completed_order_hint_list_all_no_filter() -> None:
    store = PostgresCompletedOrderHintStore(create_pg_engine(_postgres_dsn()))
    run_id = uuid.uuid4()
    ids = [uuid.uuid4() for _ in range(3)]
    for oid in ids:
        await store.add(oid, run_id=run_id)

    all_ids = await store.list_all(run_id=run_id)
    for oid in ids:
        assert oid in all_ids

    for oid in ids:
        await store.remove(oid)
