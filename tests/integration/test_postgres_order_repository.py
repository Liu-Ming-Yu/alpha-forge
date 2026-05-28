"""Integration tests for PostgresOrderRepository against a real Postgres database."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.infrastructure.postgres.repositories import (
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


def _make_intent(
    *,
    order_id: uuid.UUID | None = None,
    strategy_run_id: uuid.UUID | None = None,
    limit_price: Decimal | None = None,
) -> OrderIntent:
    now = datetime.now(tz=_UTC)
    oid = order_id or uuid.uuid4()
    srid = strategy_run_id or uuid.uuid4()
    order_type = OrderType.LIMIT if limit_price is not None else OrderType.MARKET
    intent = OrderIntent(
        order_id=oid,
        strategy_run_id=srid,
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=10,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        created_at=now,
        limit_price=limit_price,
        cash_reservation_id=uuid.uuid4(),
    )
    return intent


def _make_fill(
    *,
    order_id: uuid.UUID,
    instrument_id: uuid.UUID,
    broker_execution_id: str | None = None,
) -> FillEvent:
    now = datetime.now(tz=_UTC)
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="TEST-001",
        broker_execution_id=broker_execution_id or uuid.uuid4().hex,
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=5,
        fill_price=Decimal("150.00"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=now,
        received_at=now,
        supersedes_id=None,
    )


def _make_sell_intent_without_reservation() -> OrderIntent:
    now = datetime.now(tz=_UTC)
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        created_at=now,
        limit_price=Decimal("123.45"),
        cash_reservation_id=None,
    )


def _make_buy_intent_without_reservation() -> OrderIntent:
    now = datetime.now(tz=_UTC)
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        created_at=now,
        limit_price=Decimal("123.45"),
        cash_reservation_id=None,
    )


@pytest.mark.asyncio
async def test_save_intent_and_get_intent_round_trip() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent(limit_price=Decimal("123.45"))
    await repo.save_intent(intent)

    fetched = await repo.get_intent(intent.order_id)
    assert fetched is not None
    assert fetched.order_id == intent.order_id
    assert fetched.strategy_run_id == intent.strategy_run_id
    assert fetched.portfolio_target_id == intent.portfolio_target_id
    assert fetched.instrument_id == intent.instrument_id
    assert fetched.side == intent.side
    assert fetched.quantity == intent.quantity
    assert fetched.order_type == intent.order_type
    assert fetched.time_in_force == intent.time_in_force
    assert fetched.limit_price == intent.limit_price
    assert fetched.cash_reservation_id == intent.cash_reservation_id


@pytest.mark.asyncio
async def test_save_open_sell_intent_without_cash_reservation() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_sell_intent_without_reservation()

    await repo.save_intent(intent)

    fetched = await repo.get_intent(intent.order_id)
    assert fetched is not None
    assert fetched.side == OrderSide.SELL
    assert fetched.cash_reservation_id is None


@pytest.mark.asyncio
async def test_save_open_buy_intent_without_cash_reservation_violates_constraint() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_buy_intent_without_reservation()

    with pytest.raises(IntegrityError):
        await repo.save_intent(intent)


@pytest.mark.asyncio
async def test_get_intent_returns_none_for_unknown() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    result = await repo.get_intent(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_save_fill_deduplicates_on_broker_execution_id() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent()
    await repo.save_intent(intent)

    # Use a fresh UUID so repeated test runs don't collide on the global
    # unique index uq_fill_events_broker_execution(broker_order_id, broker_execution_id).
    exec_id = uuid.uuid4().hex
    fill1 = _make_fill(
        order_id=intent.order_id,
        instrument_id=intent.instrument_id,
        broker_execution_id=exec_id,
    )
    fill2 = FillEvent(
        fill_id=uuid.uuid4(),
        order_id=intent.order_id,
        broker_order_id="TEST-001",
        broker_execution_id=exec_id,
        instrument_id=intent.instrument_id,
        side=OrderSide.BUY,
        quantity=5,
        fill_price=Decimal("151.00"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=datetime.now(tz=_UTC),
        received_at=datetime.now(tz=_UTC),
        supersedes_id=None,
    )
    await repo.save_fill(fill1)
    await repo.save_fill(fill2)

    fills = await repo.get_fills(intent.order_id)
    assert len(fills) == 1
    assert fills[0].fill_id == fill1.fill_id


@pytest.mark.asyncio
async def test_list_open_orders_excludes_terminal() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    srid = uuid.uuid4()
    open_intent = _make_intent(strategy_run_id=srid)
    terminal_intent = _make_intent(strategy_run_id=srid)

    await repo.save_intent(open_intent)
    await repo.save_intent(terminal_intent)
    await repo.mark_terminal(terminal_intent.order_id, "test_filled")

    open_orders = await repo.list_open_orders(srid)
    ids = {o.order_id for o in open_orders}
    assert open_intent.order_id in ids
    assert terminal_intent.order_id not in ids


@pytest.mark.asyncio
async def test_mark_terminal_sets_reason() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent()
    await repo.save_intent(intent)
    await repo.mark_terminal(intent.order_id, "cancelled_by_operator")

    from sqlalchemy import text

    engine = create_pg_engine(_postgres_dsn())
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT is_terminal, terminal_reason FROM order_intents WHERE order_id = :oid"
                    ),
                    {"oid": intent.order_id},
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["is_terminal"] is True
    assert row["terminal_reason"] == "cancelled_by_operator"


@pytest.mark.asyncio
async def test_retry_transient_retries_on_class08() -> None:
    """_retry_transient decorator retries on class-08 SQLSTATE then succeeds.

    AsyncEngine.begin is a read-only property so we can't use patch.object on
    it.  Instead we replace repo._engine with a wrapper that fails on the first
    call to begin() and delegates to the real engine on subsequent calls.
    """
    real_engine = create_pg_engine(_postgres_dsn())
    repo = PostgresOrderRepository(real_engine)
    intent = _make_intent()

    call_count = 0

    class _FailOnceEngine:
        def begin(self) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FailingOnFirstEnter(real_engine)
            return real_engine.begin()

        def __getattr__(self, name: str) -> Any:
            return getattr(real_engine, name)

    class _FailingOnFirstEnter:
        def __init__(self, eng: Any) -> None:
            self._eng = eng

        async def __aenter__(self) -> None:
            from psycopg.errors import OperationalError as PsycopgOpError

            err = PsycopgOpError("connection lost")
            err.pgcode = "08006"  # type: ignore[attr-defined]
            raise OperationalError(statement=None, params=None, orig=err)

        async def __aexit__(self, *_exc: object) -> None:
            pass

    repo._engine = _FailOnceEngine()  # type: ignore[assignment]
    await repo.save_intent(intent)
    repo._engine = real_engine  # restore

    assert call_count >= 2, "expected at least one retry"
    fetched = await repo.get_intent(intent.order_id)
    assert fetched is not None


@pytest.mark.asyncio
async def test_get_fills_returns_empty_when_none() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent()
    await repo.save_intent(intent)

    fills = await repo.get_fills(intent.order_id)
    assert fills == []


@pytest.mark.asyncio
async def test_save_fill_and_get_fills_round_trip() -> None:
    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent()
    await repo.save_intent(intent)

    fill = _make_fill(order_id=intent.order_id, instrument_id=intent.instrument_id)
    await repo.save_fill(fill)

    fills = await repo.get_fills(intent.order_id)
    assert len(fills) == 1
    assert fills[0].fill_id == fill.fill_id
    assert fills[0].fill_price == fill.fill_price
    assert fills[0].quantity == fill.quantity


@pytest.mark.asyncio
async def test_record_fill_slippage_preserves_decimal_precision() -> None:
    """slippage_bps must round-trip without float precision loss.

    Regression: the prior implementation cast the Decimal slippage to float
    before binding to a NUMERIC column; small slippages compounded with
    rounding error across many fills.
    """
    from sqlalchemy import text

    repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    intent = _make_intent()
    await repo.save_intent(intent)

    fill = _make_fill(order_id=intent.order_id, instrument_id=intent.instrument_id)
    await repo.save_fill(fill)

    # 99.999999 vs 100 → |Δ|=0.000001, /100*10000 = 0.0001 bps exactly.
    expected = Decimal("100")
    actual = Decimal("99.999999")
    await repo.record_fill_slippage(fill.fill_id, expected, actual)

    async with create_pg_engine(_postgres_dsn()).connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT slippage_bps FROM fill_events WHERE fill_id = :fid"),
                    {"fid": fill.fill_id},
                )
            )
            .mappings()
            .one()
        )
    persisted = row["slippage_bps"]
    assert isinstance(persisted, Decimal)
    assert persisted == Decimal("0.0001"), f"slippage round-trip lost precision: {persisted!r}"
