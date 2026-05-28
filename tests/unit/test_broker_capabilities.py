"""Contract tests for broker capability metadata and CP stub fail-closed behavior."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.exceptions import BrokerSubmissionError
from quant_platform.infrastructure.support.simulated_broker import SimulatedBrokerGateway
from quant_platform.services.execution_service.gateways.client_portal_gateway import (
    ClientPortalBrokerGateway,
)

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)


class _Clock:
    def now(self) -> datetime:
        return _NOW

    def today(self) -> date:
        return _NOW.date()


def _snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("10000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("10000"),
        net_asset_value=Decimal("10000"),
        positions=(),
    )


def _intent() -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=Decimal("100"),
    )


@pytest.mark.asyncio
async def test_client_portal_stub_capabilities_and_fail_closed_routing() -> None:
    broker = ClientPortalBrokerGateway(clock=_Clock(), initial_snapshot=_snapshot())
    assert broker.capabilities.supports_order_routing is False
    assert broker.capabilities.supports_order_cancellation is False
    assert broker.capabilities.supports_lifecycle_feed is False

    with pytest.raises(BrokerSubmissionError) as excinfo_place:
        await broker.place_order(_intent())
    assert "does not support order routing" in str(excinfo_place.value)

    with pytest.raises(BrokerSubmissionError) as excinfo_cancel:
        await broker.cancel_order("1234")
    assert "does not support order cancellation" in str(excinfo_cancel.value)


@pytest.mark.asyncio
async def test_simulated_gateway_advertises_routing_capabilities() -> None:
    broker = SimulatedBrokerGateway(clock=_Clock(), initial_cash=Decimal("10000"))
    assert broker.capabilities.supports_order_routing is True
    assert broker.capabilities.supports_order_cancellation is True
    assert broker.capabilities.supports_lifecycle_feed is True


@pytest.mark.asyncio
async def test_simulated_gateway_place_order_is_idempotent_across_reconnect() -> None:
    broker = SimulatedBrokerGateway(clock=_Clock(), initial_cash=Decimal("10000"))
    await broker.connect()
    intent = _intent()
    ack1 = await broker.place_order(intent)
    await broker.disconnect()
    await broker.connect()
    ack2 = await broker.place_order(intent)
    assert ack1.broker_order_id == ack2.broker_order_id
