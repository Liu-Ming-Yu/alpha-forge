"""Live IBGateway order lifecycle tests: place/cancel/fill/idempotency/reconnect.

All tests place real paper orders and MUST cancel them in finally blocks.
Requires a running paper IB Gateway. Opt-in via:
  QP_LIVE_IBKR_REQUIRED=1
  QP_LIVE_IBKR_ALLOW_PAPER_ORDERS=1
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal

import pytest

from tests.ibkr_paper_safety import (
    order_client_id,
    require_env,
    require_ibapi,
    require_int_env,
    require_paper_order_safety,
    skip_unless_paper_orders_enabled,
)

pytestmark = [pytest.mark.ibapi, pytest.mark.ibapi_orders]

_UTC = UTC


def _order_setup():
    from quant_platform.config import BrokerSettings

    host = require_env("QP__BROKER__HOST")
    port = require_int_env("QP__BROKER__PORT")
    client_id = order_client_id(offset=1)
    timeout = float(require_env("QP__BROKER__REQUEST_TIMEOUT_SECONDS", default="15"))
    account_id = require_env("QP__BROKER__ACCOUNT_ID")
    require_paper_order_safety(account_id=account_id, port=port)

    con_id = require_int_env("QP__LIVE_IBKR__TEST_CON_ID")
    symbol = require_env("QP__LIVE_IBKR__TEST_SYMBOL").upper()
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-order-lifecycle:{con_id}")
    contracts = {
        instrument_id: {
            "symbol": symbol,
            "exchange": require_env("QP__LIVE_IBKR__TEST_EXCHANGE", default="SMART"),
            "currency": require_env("QP__LIVE_IBKR__TEST_CURRENCY", default="USD"),
            "con_id": con_id,
            "sec_type": "STK",
        }
    }

    settings = BrokerSettings(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        paper_trading=True,
        request_timeout_seconds=timeout,
        historical_bar_fetch_enabled=True,
    )
    return settings, contracts, instrument_id


async def _non_marketable_limit_price(broker, instrument_id: uuid.UUID) -> Decimal:
    bars = await broker.fetch_historical_bars(
        instrument_id=instrument_id,
        bar_seconds=86400,
        end_date=date.today(),
        duration="2 D",
    )
    if not bars:
        pytest.fail("no historical bars for limit price derivation")
    last_close = bars[-1].close
    return (last_close * Decimal("0.90")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _make_intent(instrument_id: uuid.UUID, limit_price: Decimal):
    from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce

    return OrderIntent(
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_limit_order_returns_ack() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack = await broker.place_order(intent)
        assert ack.order_id == intent.order_id
        assert ack.broker_order_id is not None
        assert ack.broker_order_id.isdigit() or ack.broker_order_id

        open_orders = await broker.fetch_open_orders()
        assert any(o.broker_order_id == ack.broker_order_id for o in open_orders)
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
async def test_place_order_idempotent() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack1 = await broker.place_order(intent)
        ack2 = await broker.place_order(intent)
        assert ack1.broker_order_id == ack2.broker_order_id
        ack = ack1

        open_orders = await broker.fetch_open_orders()
        matching = [o for o in open_orders if o.broker_order_id == ack1.broker_order_id]
        assert len(matching) == 1
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
async def test_place_order_unmapped_instrument_raises() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
    from quant_platform.core.exceptions import BrokerSubmissionError
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, _ = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    await broker.connect()
    try:
        unmapped_id = uuid.uuid4()
        intent = OrderIntent(
            order_id=uuid.uuid4(),
            strategy_run_id=uuid.uuid4(),
            portfolio_target_id=uuid.uuid4(),
            instrument_id=unmapped_id,
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("1.00"),
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(tz=_UTC),
        )
        with pytest.raises(BrokerSubmissionError):
            await broker.place_order(intent)
    finally:
        await broker.disconnect()


@pytest.mark.asyncio
async def test_cancel_order_drains_lifecycle_event() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.core.domain.orders.lifecycle import BrokerOrderCancelled
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    cancelled = False
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack = await broker.place_order(intent)
        await broker.cancel_order(ack.broker_order_id)
        cancelled = True
        await asyncio.sleep(1)

        lifecycle_events = await broker.drain_lifecycle_events()
        cancelled_events = [
            e
            for e in lifecycle_events
            if isinstance(e, BrokerOrderCancelled) and e.order_id == intent.order_id
        ]
        assert len(cancelled_events) >= 1
    finally:
        if ack is not None and not cancelled:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
async def test_cancel_order_removed_from_open_orders() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    cancelled = False
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack = await broker.place_order(intent)
        await broker.cancel_order(ack.broker_order_id)
        cancelled = True
        await asyncio.sleep(2)

        open_orders = await broker.fetch_open_orders()
        assert all(o.broker_order_id != ack.broker_order_id for o in open_orders)
    finally:
        if ack is not None and not cancelled:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
async def test_drain_lifecycle_events_clears_queue() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack = await broker.place_order(intent)
        await broker.cancel_order(ack.broker_order_id)
        await asyncio.sleep(1)

        first_drain = await broker.drain_lifecycle_events()
        assert len(first_drain) >= 1

        second_drain = await broker.drain_lifecycle_events()
        assert second_drain == []
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()


@pytest.mark.asyncio
async def test_place_order_reconnect_no_duplicate() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
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
async def test_fetch_open_orders_rebuilds_lifecycle_maps() -> None:
    skip_unless_paper_orders_enabled()
    require_ibapi()
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )

    settings, contracts, instrument_id = _order_setup()
    broker = IBGatewayBrokerGateway(settings=settings, instrument_contracts=contracts)
    ack = None
    await broker.connect()
    try:
        limit_price = await _non_marketable_limit_price(broker, instrument_id)
        intent = _make_intent(instrument_id, limit_price)
        ack = await broker.place_order(intent)
        ib_order_id = int(ack.broker_order_id)

        await broker.disconnect()
        await broker.connect()

        assert ib_order_id in broker._wrapper._ib_to_internal
        assert ib_order_id in broker._wrapper._ib_to_instrument
    finally:
        if ack is not None:
            with suppress(Exception):
                await broker.cancel_order(ack.broker_order_id)
        await broker.disconnect()
