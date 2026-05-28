"""End-to-end tests: IBGateway + Postgres + Redis all wired together.

Tests that require IBGateway are marked `ibapi` and `e2e`.
Tests that only need Postgres + Redis are marked `integration_durable` and `e2e`.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

import pytest

from tests.ibkr_paper_safety import (
    paper_orders_enabled,
    require_paper_order_safety,
)

_UTC = UTC


def _live_enabled() -> bool:
    return (
        os.environ.get("QP_LIVE_IBKR_REQUIRED", "").strip() == "1"
        or os.environ.get("QP_VERIFY_LIVE_IBKR", "").strip() == "1"
    )


def _orders_enabled() -> bool:
    return paper_orders_enabled()


def _require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if _live_enabled():
        pytest.fail(f"{name} is required for e2e tests")
    pytest.skip(f"{name} is not configured")


def _require_int_env(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer")


def _require_ibapi() -> None:
    try:
        __import__("ibapi")
    except Exception as exc:
        if _live_enabled():
            pytest.fail(f"ibapi is required: {exc}")
        pytest.skip(f"ibapi is not installed: {exc}")


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for e2e tests")
    return dsn


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for e2e tests")
    return url


def _skip_unless_live_orders() -> None:
    if not _live_enabled():
        pytest.skip("set QP_LIVE_IBKR_REQUIRED=1 for e2e live tests")
    if not _orders_enabled():
        pytest.skip("set QP_LIVE_IBKR_ALLOW_PAPER_ORDERS=1 for e2e order tests")


def _order_instrument():
    symbol = os.environ.get("QP__LIVE_IBKR__TEST_SYMBOL", "").strip().upper()
    con_id_raw = os.environ.get("QP__LIVE_IBKR__TEST_CON_ID", "").strip()
    if not symbol or not con_id_raw:
        pytest.skip("QP__LIVE_IBKR__TEST_SYMBOL and QP__LIVE_IBKR__TEST_CON_ID required")
    con_id = int(con_id_raw)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-e2e-cycle:{con_id}")
    spec = {
        "symbol": symbol,
        "exchange": os.environ.get("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART"),
        "currency": os.environ.get("QP__LIVE_IBKR__TEST_CURRENCY", "USD"),
        "con_id": con_id,
        "sec_type": "STK",
    }
    return instrument_id, spec


# ---------------------------------------------------------------------------
# E2E: broker + Postgres + Redis full cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.ibapi
@pytest.mark.ibapi_orders
@pytest.mark.e2e
async def test_full_cycle_broker_order_persisted_and_event_via_redis() -> None:
    """Wire IBGateway + PostgresOrderRepository + RedisStreamsEventBus.

    Place a non-marketable GTC limit order, drain lifecycle, assert order saved
    in Postgres, assert OrderSubmitted event delivered via Redis Streams.
    """
    _skip_unless_live_orders()
    _require_ibapi()

    from quant_platform.config import BrokerSettings
    from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
    from quant_platform.core.events import OrderSubmitted
    from quant_platform.infrastructure.event_bus import RedisStreamsEventBus
    from quant_platform.infrastructure.postgres.repositories import (
        PostgresOrderRepository,
        create_pg_engine,
    )
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _require_int_env("QP__BROKER__CLIENT_ID") + 3
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    timeout = float(_require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="15"))
    require_paper_order_safety(account_id=account_id, port=port)

    instrument_id, spec = _order_instrument()
    contracts = {instrument_id: spec}

    broker_settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=True,
    )
    broker = IBGatewayBrokerGateway(settings=broker_settings, instrument_contracts=contracts)
    order_repo = PostgresOrderRepository(create_pg_engine(_postgres_dsn()))
    prefix = f"qp:test:e2e:{uuid.uuid4()}"
    bus = RedisStreamsEventBus(
        redis_url=_redis_url(),
        stream_prefix=prefix,
        block_ms=100,
        use_consumer_groups=True,
    )

    ack = None
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=86400,
            end_date=date.today(),
            duration="2 D",
        )
        if not bars:
            pytest.skip("no bars available for limit price derivation")
        limit_price = (bars[-1].close * Decimal("0.90")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        intent = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=uuid.uuid4(),
            portfolio_target_id=uuid.uuid4(),
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(tz=_UTC),
            cash_reservation_id=uuid.uuid4(),
        )

        await order_repo.save_intent(intent)
        ack = await broker.place_order(intent)

        await bus.publish(
            OrderSubmitted(
                event_id=uuid.uuid4(),
                occurred_at=datetime.now(tz=_UTC),
                order_id=intent.order_id,
                broker_order_id=ack.broker_order_id,
            )
        )

        fetched = await order_repo.get_intent(intent.order_id)
        assert fetched is not None
        assert fetched.order_id == intent.order_id

        consumer = bus.subscribe(OrderSubmitted, f"e2e-consumer-{uuid.uuid4()}")
        received = await asyncio.wait_for(anext(consumer), timeout=5)
        assert received.order_id == intent.order_id
        await consumer.aclose()
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
@pytest.mark.ibapi
@pytest.mark.ibapi_orders
@pytest.mark.e2e
async def test_reconnect_no_duplicate_orders_live() -> None:
    """Place a GTC order, disconnect/reconnect — idempotency guard fires, count stays 1."""
    _skip_unless_live_orders()
    _require_ibapi()

    from quant_platform.config import BrokerSettings
    from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _require_int_env("QP__BROKER__CLIENT_ID") + 4
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    timeout = float(_require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="15"))
    require_paper_order_safety(account_id=account_id, port=port)

    instrument_id, spec = _order_instrument()
    broker_settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=True,
    )
    broker = IBGatewayBrokerGateway(
        settings=broker_settings, instrument_contracts={instrument_id: spec}
    )
    ack = None
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=86400,
            end_date=date.today(),
            duration="2 D",
        )
        if not bars:
            pytest.skip("no bars for limit price")
        limit_price = (bars[-1].close * Decimal("0.90")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        intent = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=uuid.uuid4(),
            portfolio_target_id=uuid.uuid4(),
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(tz=_UTC),
        )
        ack = await broker.place_order(intent)

        await broker.disconnect()
        await broker.connect()

        assert intent.order_id in broker._submitted
        ack2 = await broker.place_order(intent)
        assert ack2.broker_order_id == ack.broker_order_id

        open_orders = await broker.fetch_open_orders()
        matching = [o for o in open_orders if o.broker_order_id == ack.broker_order_id]
        assert len(matching) == 1
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
@pytest.mark.ibapi
@pytest.mark.e2e
async def test_kill_switch_persists_across_broker_reconnect() -> None:
    """Activate PostgresKillSwitchStore; after reconnect hydrate_session_state confirms active."""
    if not _live_enabled():
        pytest.skip("set QP_LIVE_IBKR_REQUIRED=1")
    _require_ibapi()

    from quant_platform.config import BrokerSettings
    from quant_platform.infrastructure.postgres.repositories import create_pg_engine
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )
    from quant_platform.services.execution_service.stores.kill_switch_store import (
        PostgresKillSwitchStore,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _require_int_env("QP__BROKER__CLIENT_ID") + 5
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")

    broker_settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
    )
    broker = IBGatewayBrokerGateway(settings=broker_settings, instrument_contracts={})
    ks_store = PostgresKillSwitchStore(create_pg_engine(_postgres_dsn()))
    now = datetime.now(tz=_UTC)

    await ks_store.activate(reason="e2e_kill_switch_test", activated_by="pytest_e2e", as_of=now)
    try:
        await broker.connect()
        await broker.disconnect()
        await broker.connect()
        try:
            state = await ks_store.get()
            assert state.active is True
            assert state.reason == "e2e_kill_switch_test"
        finally:
            await broker.disconnect()
    finally:
        await ks_store.clear(operator_id="pytest_cleanup", as_of=datetime.now(tz=_UTC))


@pytest.mark.asyncio
@pytest.mark.integration_durable
@pytest.mark.e2e
async def test_settlement_lot_advance_to_t2_with_postgres() -> None:
    """Seed a settlement lot due yesterday; advance_settlements deletes it."""
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.settlement import SettlementLot, SettlementStatus
    from quant_platform.infrastructure.event_bus import InMemoryEventBus
    from quant_platform.infrastructure.postgres.repositories import create_pg_engine
    from quant_platform.infrastructure.repositories import InMemoryOrderRepository
    from quant_platform.infrastructure.support.clock import WallClock
    from quant_platform.services.execution_service.account.account_state_coordinator import (
        AccountStateCoordinator,
    )
    from quant_platform.services.execution_service.stores.pending_settlement_store import (
        InMemoryCompletedOrderHintStore,
        PostgresPendingSettlementStore,
    )
    from quant_platform.services.portfolio_service.cash_ledger import CashLedger
    from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

    engine = create_pg_engine(_postgres_dsn())
    settlement_store = PostgresPendingSettlementStore(engine)
    event_bus = InMemoryEventBus()
    order_repo = InMemoryOrderRepository()
    hint_store = InMemoryCompletedOrderHintStore()

    run_id = uuid.uuid4()
    order_id = uuid.uuid4()
    now = datetime.now(tz=UTC)

    initial_snapshot = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=now,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("1000"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("101000"),
        positions=(),
    )
    clock = WallClock()
    cash_engine = CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=initial_snapshot,
    )

    yesterday = date.today() - timedelta(days=1)
    lot = SettlementLot(
        lot_id=uuid.uuid4(),
        fill_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        trade_date=yesterday - timedelta(days=2),
        settlement_date=yesterday,
        gross_proceeds=Decimal("1001.00"),
        commission=Decimal("1.00"),
        net_proceeds=Decimal("1000.00"),
        currency="USD",
        status=SettlementStatus.PENDING,
    )
    await settlement_store.upsert(lot, run_id=run_id, order_id=order_id)

    coordinator = AccountStateCoordinator(
        cash_engine=cash_engine,
        event_bus=event_bus,
        clock=clock,
        strategy_run_id=run_id,
        order_repo=order_repo,
        pending_settlement_store=settlement_store,
        completed_order_hint_store=hint_store,
    )
    await coordinator.hydrate()
    advanced = await coordinator.advance_settlements()
    assert advanced >= 1

    remaining = await settlement_store.list_all(run_id=run_id)
    assert not any(remaining_lot.lot_id == lot.lot_id for remaining_lot in remaining)


@pytest.mark.asyncio
@pytest.mark.integration_durable
@pytest.mark.e2e
async def test_governance_preflight_with_real_services() -> None:
    """evaluate_preflight passes postgres_configured and redis_configured checks."""
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import ProductionProfile
    from quant_platform.services.governance_service.preflight import evaluate_preflight

    dsn = _postgres_dsn()
    redis = _redis_url()

    settings = PlatformSettings(
        _env_file=None,
    )
    settings.storage.postgres_dsn = dsn
    settings.storage.redis_url = redis
    settings.storage.event_bus_backend = "redis_streams"
    settings.api.operator_api_key = "test-key"
    settings.api.allow_unauthenticated = False
    settings.api.acknowledge_unauthenticated_risk = False

    report = evaluate_preflight(settings, profile=ProductionProfile.PAPER)
    check_map = {c.name: c for c in report.checks}

    assert check_map["postgres_configured"].passed
    assert check_map["redis_configured"].passed
    assert check_map["redis_streams_enabled"].passed


@pytest.mark.asyncio
@pytest.mark.integration_durable
@pytest.mark.e2e
async def test_operator_api_all_endpoints_with_real_repos() -> None:
    """Start FastAPI with real Postgres repos; all read endpoints return 200."""
    try:
        import httpx
        import starlette.testclient  # noqa: F401
    except ImportError:
        pytest.skip("httpx/starlette required for API e2e test")

    from quant_platform.infrastructure.event_bus import InMemoryEventBus
    from quant_platform.infrastructure.performance import PostgresPerformanceRepository
    from quant_platform.infrastructure.postgres.repositories import (
        PostgresAuditSink,
        PostgresOrderRepository,
        PostgresPositionRepository,
        create_pg_engine,
    )
    from quant_platform.views.operator_api.app import create_app

    engine = create_pg_engine(_postgres_dsn())
    order_repo = PostgresOrderRepository(engine)
    position_repo = PostgresPositionRepository(engine)
    performance_repo = PostgresPerformanceRepository(engine)
    audit_sink = PostgresAuditSink(engine)
    event_bus = InMemoryEventBus()

    api_key = "e2e-test-key-" + uuid.uuid4().hex[:8]
    from quant_platform.config import PlatformSettings

    settings = PlatformSettings(_env_file=None)
    settings.api.operator_api_key = api_key

    app = create_app(
        settings=settings,
        order_repo=order_repo,
        position_repo=position_repo,
        performance_repo=performance_repo,
        event_bus=event_bus,
        audit_sink=audit_sink,
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200

            resp = await client.get(
                "/cash",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200

            resp = await client.get(
                f"/blotter/{uuid.uuid4()}",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200

            resp = await client.get(
                "/audit",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
    except ImportError:
        pytest.skip("ASGITransport not available in this httpx version")
