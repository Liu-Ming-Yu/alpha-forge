"""Opt-in paper IBKR order-routing smoke test.

This test intentionally submits a real order to a paper IB Gateway/TWS session.
It is separate from the read-only live smoke so normal ``-m ibapi`` verification
does not place orders unless the operator explicitly opts in.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from tests.ibkr_paper_safety import (
    order_client_id,
    require_env,
    require_ibapi,
    require_int_env,
    require_paper_order_safety,
    skip_unless_paper_orders_enabled,
)

pytestmark = pytest.mark.ibapi

InstrumentContracts = dict[uuid.UUID, dict[str, object]]


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


def _order_contract() -> tuple[InstrumentContracts, uuid.UUID]:
    contracts_path = Path(require_env("QP__LIVE_IBKR__CONTRACTS_FILE"))
    contracts = _load_contracts_file(contracts_path)

    con_id = require_int_env("QP__LIVE_IBKR__TEST_CON_ID")
    for instrument_id, spec in contracts.items():
        if spec.get("con_id") == con_id:
            return contracts, instrument_id

    symbol = require_env("QP__LIVE_IBKR__TEST_SYMBOL").upper()
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-live-paper-order:{con_id}")
    contracts[instrument_id] = {
        "symbol": symbol,
        "exchange": require_env("QP__LIVE_IBKR__TEST_EXCHANGE", default="SMART"),
        "currency": require_env("QP__LIVE_IBKR__TEST_CURRENCY", default="USD"),
        "con_id": con_id,
        "sec_type": "STK",
    }
    return contracts, instrument_id


async def _non_marketable_limit_price(broker: object, instrument_id: uuid.UUID) -> Decimal:
    bars = await broker.fetch_historical_bars(
        instrument_id=instrument_id,
        bar_seconds=86400,
        end_date=date.today(),
        duration="2 D",
    )
    if not bars:
        pytest.fail("could not derive paper order limit price: no historical bars returned")
    latest_close = bars[-1].close
    return (latest_close * Decimal("0.90")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@pytest.mark.asyncio
async def test_live_ib_gateway_paper_limit_order_place_and_cancel() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()

    from quant_platform.config import BrokerSettings
    from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
    from quant_platform.core.domain.orders.lifecycle import BrokerOrderCancelled
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    host = require_env("QP__BROKER__HOST")
    port = require_int_env("QP__BROKER__PORT")
    client_id = order_client_id(offset=1)
    timeout = float(require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="10"))
    account_id = require_env("QP__BROKER__ACCOUNT_ID")
    require_paper_order_safety(account_id=account_id, port=port)
    contracts, instrument_id = _order_contract()

    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=True,
    )
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    cancelled = False

    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
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
            created_at=datetime.now(tz=UTC),
        )

        ack = await broker.place_order(intent)
        assert ack.order_id == intent.order_id
        assert ack.broker_order_id is not None

        await broker.cancel_order(ack.broker_order_id)
        cancelled = True
        await asyncio.sleep(1)

        lifecycle_events = await broker.drain_lifecycle_events()
        assert any(
            isinstance(event, BrokerOrderCancelled) and event.order_id == intent.order_id
            for event in lifecycle_events
        )

        open_orders = await broker.fetch_open_orders()
        assert all(order.broker_order_id != ack.broker_order_id for order in open_orders)
    finally:
        if ack is not None and ack.broker_order_id is not None and not cancelled:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()
