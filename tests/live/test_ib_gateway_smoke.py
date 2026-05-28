"""Read-only live smoke tests for a paper IB Gateway / TWS session."""

from __future__ import annotations

import json
import os
import uuid
from datetime import date
from pathlib import Path

import pytest

pytestmark = pytest.mark.ibapi

InstrumentContracts = dict[uuid.UUID, dict[str, object]]


def _live_enabled() -> bool:
    return (
        os.environ.get("QP_LIVE_IBKR_REQUIRED", "").strip() == "1"
        or os.environ.get("QP_VERIFY_LIVE_IBKR", "").strip() == "1"
    )


def _skip_unless_live_enabled() -> None:
    if not _live_enabled():
        pytest.skip(
            "live IBKR smoke tests are opt-in; set QP_LIVE_IBKR_REQUIRED=1 or QP_VERIFY_LIVE_IBKR=1"
        )


def _require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if _live_enabled():
        pytest.fail(f"{name} is required for live IBKR smoke tests")
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
            pytest.fail(f"ibapi is required for live IBKR smoke tests: {exc}")
        pytest.skip(f"ibapi is not installed: {exc}")


def _contracts_file_path() -> Path | None:
    raw = os.environ.get("QP__LIVE_IBKR__CONTRACTS_FILE", "").strip()
    if not raw:
        return None
    return Path(raw)


def _load_contracts_file(path: Path) -> InstrumentContracts:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        pytest.fail(f"could not read QP__LIVE_IBKR__CONTRACTS_FILE={path}: {exc}")
    except json.JSONDecodeError as exc:
        pytest.fail(f"invalid JSON in QP__LIVE_IBKR__CONTRACTS_FILE={path}: {exc}")

    if not isinstance(payload, dict):
        pytest.fail("QP__LIVE_IBKR__CONTRACTS_FILE must contain a JSON object")

    contracts: InstrumentContracts = {}
    for raw_id, raw_spec in payload.items():
        try:
            instrument_id = uuid.UUID(str(raw_id))
        except ValueError:
            pytest.fail(f"invalid instrument UUID in contracts file: {raw_id!r}")
        if not isinstance(raw_spec, dict):
            pytest.fail(f"contract spec for {instrument_id} must be a JSON object")
        spec = dict(raw_spec)
        con_id = spec.get("con_id")
        if not (isinstance(con_id, int) and con_id > 0):
            pytest.fail(f"contract spec for {instrument_id} must include numeric con_id")
        contracts[instrument_id] = spec
    return contracts


def _optional_contract() -> tuple[uuid.UUID, dict[str, object]] | None:
    symbol = os.environ.get("QP__LIVE_IBKR__TEST_SYMBOL", "").strip().upper()
    con_id_raw = os.environ.get("QP__LIVE_IBKR__TEST_CON_ID", "").strip()
    if not symbol and not con_id_raw:
        return None
    if not symbol or not con_id_raw:
        if _live_enabled():
            pytest.fail(
                "QP__LIVE_IBKR__TEST_SYMBOL and QP__LIVE_IBKR__TEST_CON_ID "
                "must be supplied together"
            )
        pytest.skip("live IBKR historical-bar smoke contract is incomplete")
    try:
        con_id = int(con_id_raw)
    except ValueError:
        pytest.fail("QP__LIVE_IBKR__TEST_CON_ID must be an integer")
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-live-smoke:{con_id}")
    exchange = _require_env("QP__LIVE_IBKR__TEST_EXCHANGE", default="SMART")
    currency = _require_env("QP__LIVE_IBKR__TEST_CURRENCY", default="USD")
    return instrument_id, {
        "symbol": symbol,
        "exchange": exchange,
        "currency": currency,
        "con_id": con_id,
    }


def _optional_contracts() -> tuple[InstrumentContracts, uuid.UUID | None]:
    contracts: InstrumentContracts = {}
    contracts_path = _contracts_file_path()
    if contracts_path is not None:
        contracts.update(_load_contracts_file(contracts_path))

    historical_contract = _optional_contract()
    historical_instrument_id: uuid.UUID | None = None
    if historical_contract is not None:
        historical_instrument_id, spec = historical_contract
        con_id = spec["con_id"]
        for existing_id, existing_spec in contracts.items():
            if existing_spec.get("con_id") == con_id:
                historical_instrument_id = existing_id
                break
        else:
            contracts[historical_instrument_id] = spec

    return contracts, historical_instrument_id


@pytest.mark.asyncio
async def test_live_ib_gateway_read_only_smoke() -> None:
    _skip_unless_live_enabled()
    _require_ibapi()

    from quant_platform.config import BrokerSettings
    from quant_platform.core.contracts import BrokerHealthStatus
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = _require_env("QP__BROKER__HOST")
    port = _require_int_env("QP__BROKER__PORT")
    client_id = _require_int_env("QP__BROKER__CLIENT_ID") + 100
    timeout = float(_require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="10"))
    account_id = _require_env("QP__BROKER__ACCOUNT_ID")
    contracts, historical_instrument_id = _optional_contracts()

    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=historical_instrument_id is not None,
    )
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)

    await broker.connect()
    try:
        health = await broker.health_check()
        assert health.status == BrokerHealthStatus.CONNECTED

        open_orders = await broker.fetch_open_orders()
        assert isinstance(open_orders, list)

        snapshot = await broker.sync_account()
        assert snapshot.source == "broker"
        assert snapshot.as_of is not None
        assert snapshot.net_asset_value >= 0

        positions = await broker.sync_positions()
        assert isinstance(positions, list)

        lifecycle_events = await broker.drain_lifecycle_events()
        assert isinstance(lifecycle_events, list)

        if historical_instrument_id is not None:
            bars = await broker.fetch_historical_bars(
                instrument_id=historical_instrument_id,
                bar_seconds=86400,
                end_date=date.today(),
                duration="2 D",
            )
            assert isinstance(bars, list)
            assert all(bar.instrument_id == historical_instrument_id for bar in bars)
    finally:
        await broker.disconnect()
