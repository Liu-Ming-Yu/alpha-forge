"""Live IBGateway lifecycle tests: connect/disconnect/health/account/positions/bars.

All tests are read-only (no orders placed). Requires a running paper IB Gateway
or TWS. Opt-in: set QP_LIVE_IBKR_REQUIRED=1.
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from decimal import Decimal
from itertools import count

import pytest

pytestmark = pytest.mark.ibapi

_CLIENT_ID_OFFSETS = count(0)


# ---------------------------------------------------------------------------
# Guard helpers (duplicated from test_ib_gateway_smoke.py to keep files
# self-contained; extract to tests/live/helpers.py in a future refactor)
# ---------------------------------------------------------------------------


def _live_enabled() -> bool:
    return (
        os.environ.get("QP_LIVE_IBKR_REQUIRED", "").strip() == "1"
        or os.environ.get("QP_VERIFY_LIVE_IBKR", "").strip() == "1"
    )


def _skip_unless_live_enabled() -> None:
    if not _live_enabled():
        pytest.skip("live IBKR tests are opt-in; set QP_LIVE_IBKR_REQUIRED=1")


def _require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if _live_enabled():
        pytest.fail(f"{name} is required for live IBKR tests")
    pytest.skip(f"{name} is not configured")


def _require_int_env(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer")


def _next_client_id(*, offset: int = 0) -> int:
    return _require_int_env("QP__BROKER__CLIENT_ID") + offset + next(_CLIENT_ID_OFFSETS)


def _require_ibapi() -> None:
    try:
        __import__("ibapi")
    except Exception as exc:
        if _live_enabled():
            pytest.fail(f"ibapi is required for live IBKR tests: {exc}")
        pytest.skip(f"ibapi is not installed: {exc}")


def _broker_settings():
    from quant_platform.config import BrokerSettings

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _next_client_id()
    timeout = float(_require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="10"))
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    return BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=True,
    )


def _test_instrument() -> tuple[uuid.UUID, dict] | None:
    symbol = os.environ.get("QP__LIVE_IBKR__TEST_SYMBOL", "").strip().upper()
    con_id_raw = os.environ.get("QP__LIVE_IBKR__TEST_CON_ID", "").strip()
    if not symbol or not con_id_raw:
        return None
    con_id = int(con_id_raw)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-lifecycle:{con_id}")
    return instrument_id, {
        "symbol": symbol,
        "exchange": os.environ.get("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART"),
        "currency": os.environ.get("QP__LIVE_IBKR__TEST_CURRENCY", "USD"),
        "con_id": con_id,
        "sec_type": "STK",
    }


# ---------------------------------------------------------------------------
# Connect / disconnect / reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_sets_connected_flag() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    assert broker._connected is False

    await broker.connect()
    try:
        assert broker._connected is True
    finally:
        await broker.disconnect()

    assert broker._connected is False


@pytest.mark.asyncio
async def test_connect_is_idempotent() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})

    await broker.connect()
    try:
        await broker.connect()
        assert broker._connected is True
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_reconnect_after_disconnect_restores_health() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.core.contracts import BrokerHealthStatus
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})

    await broker.connect()
    await broker.disconnect()
    await broker.connect()
    try:
        health = await broker.health_check()
        assert health.status == BrokerHealthStatus.CONNECTED
    finally:
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_connected() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.core.contracts import BrokerHealthStatus
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        health = await broker.health_check()
        assert health.status == BrokerHealthStatus.CONNECTED
        assert (
            health.latency_ms >= 0.0
        )  # loopback on Windows resolves in <1 timer tick (0.0 is valid)
        assert health.latency_ms < settings.request_timeout_seconds * 1000 + 100
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_health_check_returns_degraded_on_tight_timeout() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.config import BrokerSettings
    from quant_platform.core.contracts import BrokerHealthStatus
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _next_client_id(offset=50)
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    tight_settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=0.001,
    )
    broker = IBGatewayBrokerGateway(settings=tight_settings, instrument_contracts={})
    try:
        await broker.connect()
    except Exception:
        pytest.skip("could not connect with tight timeout settings — skipping")
    try:
        health = await broker.health_check()
        assert health.status == BrokerHealthStatus.DEGRADED
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_health_check_before_connect_returns_disconnected() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.core.contracts import BrokerHealthStatus
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    health = await broker.health_check()
    assert health.status == BrokerHealthStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Account sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_account_returns_valid_snapshot() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        snapshot = await broker.sync_account()
        assert snapshot.source == "broker"
        assert snapshot.as_of is not None
        assert snapshot.settled_cash >= Decimal("0")
        assert snapshot.net_asset_value >= Decimal("0")
        assert isinstance(snapshot.positions, tuple)
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_sync_account_cash_fields_consistent() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        snapshot = await broker.sync_account()
        assert snapshot.settled_cash >= Decimal("0")
        assert snapshot.unsettled_cash >= Decimal("0")
        assert snapshot.reserved_cash >= Decimal("0")
        expected_available = snapshot.settled_cash - snapshot.reserved_cash
        assert abs(snapshot.available_cash - expected_available) <= Decimal("0.01")
    finally:
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Position sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_positions_returns_list() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        positions = await broker.sync_positions()
        assert isinstance(positions, list)
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_sync_positions_skips_zero_quantity() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        positions = await broker.sync_positions()
        for pos in positions:
            assert pos.quantity > 0
    finally:
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Open orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_list() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        open_orders = await broker.fetch_open_orders()
        assert isinstance(open_orders, list)
    finally:
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Historical bars
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_historical_bars_daily() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    instrument = _test_instrument()
    if instrument is None:
        pytest.skip("QP__LIVE_IBKR__TEST_CON_ID / QP__LIVE_IBKR__TEST_SYMBOL not set")

    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    instrument_id, spec = instrument
    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={instrument_id: spec})
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=86400,
            end_date=date.today(),
            duration="5 D",
        )
        assert isinstance(bars, list)
        if bars:
            for bar in bars:
                assert bar.instrument_id == instrument_id
                assert bar.bar_seconds == 86400
                assert bar.open > Decimal("0")
                assert bar.close > Decimal("0")
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_fetch_historical_bars_1min() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    instrument = _test_instrument()
    if instrument is None:
        pytest.skip("QP__LIVE_IBKR__TEST_CON_ID / QP__LIVE_IBKR__TEST_SYMBOL not set")

    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    instrument_id, spec = instrument
    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={instrument_id: spec})
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=60,
            end_date=date.today(),
            duration="1 D",
        )
        assert isinstance(bars, list)
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_fetch_historical_bars_5min() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    instrument = _test_instrument()
    if instrument is None:
        pytest.skip("QP__LIVE_IBKR__TEST_CON_ID / QP__LIVE_IBKR__TEST_SYMBOL not set")

    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    instrument_id, spec = instrument
    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={instrument_id: spec})
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=300,
            end_date=date.today(),
            duration="1 D",
        )
        assert isinstance(bars, list)
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_fetch_historical_bars_disabled_returns_empty() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _next_client_id()
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        historical_bar_fetch_enabled=False,
    )
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    bars = await broker.fetch_historical_bars(
        instrument_id=uuid.uuid4(),
        bar_seconds=86400,
        end_date=date.today(),
        duration="1 D",
    )
    assert bars == []


@pytest.mark.asyncio
async def test_fetch_historical_bars_unmapped_instrument_returns_empty() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings = _broker_settings()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        bars = await broker.fetch_historical_bars(
            instrument_id=uuid.uuid4(),
            bar_seconds=86400,
            end_date=date.today(),
            duration="1 D",
        )
        assert bars == []
    finally:
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Instrument mapping validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_instrument_mappings_warns_missing_con_id() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=_next_client_id(),
        account_id=account_id,
        paper_trading=True,
    )
    missing_con_id = uuid.uuid4()
    broker = IBGatewayBrokerGateway(
        settings=settings,
        instrument_contracts={
            missing_con_id: {"symbol": "TEST", "exchange": "SMART", "currency": "USD"}
        },
    )
    warnings = broker.validate_instrument_mappings()
    assert len(warnings) >= 1
    assert any("con_id" in w.lower() or str(missing_con_id) in w for w in warnings)


@pytest.mark.asyncio
async def test_validate_instrument_mappings_passes_all_con_id() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=_next_client_id(),
        account_id=account_id,
        paper_trading=True,
    )
    instrument_id = uuid.uuid4()
    broker = IBGatewayBrokerGateway(
        settings=settings,
        instrument_contracts={
            instrument_id: {
                "symbol": "SPY",
                "exchange": "SMART",
                "currency": "USD",
                "con_id": 756733,
                "sec_type": "STK",
            }
        },
    )
    warnings = broker.validate_instrument_mappings()
    assert warnings == []
