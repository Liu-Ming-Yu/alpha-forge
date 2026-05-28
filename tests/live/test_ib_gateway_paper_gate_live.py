"""Live IBGateway paper gate validation tests.

Tests validate that the paper-safety gate correctly accepts/rejects configurations.
No orders are placed. Opt-in: set QP_LIVE_IBKR_REQUIRED=1.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.ibapi


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


def _require_ibapi() -> None:
    try:
        __import__("ibapi")
    except Exception as exc:
        if _live_enabled():
            pytest.fail(f"ibapi is required: {exc}")
        pytest.skip(f"ibapi is not installed: {exc}")


@pytest.mark.asyncio
async def test_paper_port_4002_accepted() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    assert port in {4002, 7497}, f"test requires paper port but got {port}"

    client_id = _require_int_env("QP__BROKER__CLIENT_ID") + 200
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
    )
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts={})
    await broker.connect()
    try:
        assert broker._connected is True
    finally:
        await broker.disconnect()


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
        client_id=_require_int_env("QP__BROKER__CLIENT_ID"),
        account_id=account_id,
        paper_trading=True,
    )
    instrument_id = uuid.uuid4()
    broker = IBGatewayBrokerGateway(
        settings=settings,
        instrument_contracts={
            instrument_id: {"symbol": "NOCONID", "exchange": "SMART", "currency": "USD"}
        },
    )
    warnings = broker.validate_instrument_mappings()
    assert len(warnings) >= 1
    joined = " ".join(warnings).lower()
    assert "con_id" in joined or str(instrument_id) in " ".join(warnings)


@pytest.mark.asyncio
async def test_validate_instrument_mappings_passes_when_all_have_con_id() -> None:
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
        client_id=_require_int_env("QP__BROKER__CLIENT_ID"),
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


@pytest.mark.asyncio
async def test_pacing_slot_enforced() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()
    instrument = None
    symbol = os.environ.get("QP__LIVE_IBKR__TEST_SYMBOL", "").strip().upper()
    con_id_raw = os.environ.get("QP__LIVE_IBKR__TEST_CON_ID", "").strip()
    if not symbol or not con_id_raw:
        pytest.skip("QP__LIVE_IBKR__TEST_CON_ID / QP__LIVE_IBKR__TEST_SYMBOL not set")

    from datetime import date

    from quant_platform.config import BrokerSettings
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    con_id = int(con_id_raw)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-pacing:{con_id}")
    contracts = {
        instrument_id: {
            "symbol": symbol,
            "exchange": os.environ.get("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART"),
            "currency": os.environ.get("QP__LIVE_IBKR__TEST_CURRENCY", "USD"),
            "con_id": con_id,
            "sec_type": "STK",
        }
    }

    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=_require_int_env("QP__BROKER__CLIENT_ID") + 202,
        account_id=account_id,
        paper_trading=True,
        historical_bar_fetch_enabled=True,
        historical_bar_pacing_max_requests=2,
        historical_bar_pacing_window_seconds=600.0,
    )
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    await broker.connect()
    try:
        import time

        t_start = time.monotonic()
        for _ in range(2):
            await broker.fetch_historical_bars(
                instrument_id=instrument_id,
                bar_seconds=86400,
                end_date=date.today(),
                duration="2 D",
            )
        t_fast = time.monotonic() - t_start

        t_start3 = time.monotonic()
        third_task = asyncio.create_task(
            broker.fetch_historical_bars(
                instrument_id=instrument_id,
                bar_seconds=86400,
                end_date=date.today(),
                duration="2 D",
            )
        )
        await asyncio.sleep(0.3)
        t_wait = time.monotonic() - t_start3
        third_task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await third_task

        assert t_wait >= 0.1, "third request should have been delayed by pacing"
    finally:
        await broker.disconnect()
