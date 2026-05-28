"""Tests for approval controller reservation semantics."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.events import OrderApproved
from quant_platform.infrastructure.event_bus import InMemoryEventBus
from quant_platform.infrastructure.repositories import InMemoryOrderRepository
from quant_platform.services.portfolio_service.controllers import ApproveOrdersControllerImpl

_NOW = datetime(2026, 5, 18, 14, 0, tzinfo=UTC)
_INSTRUMENT = uuid.uuid4()


def _account_with_position() -> AccountSnapshot:
    position = PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=_INSTRUMENT,
        quantity=25,
        average_cost=Decimal("100"),
        market_price=Decimal("110"),
        market_value=Decimal("2750"),
        unrealised_pnl=Decimal("250"),
        as_of=_NOW,
    )
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("102750"),
        positions=(position,),
    )


def _sell_intent() -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=_INSTRUMENT,
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=Decimal("109"),
    )


class _PassingGate:
    def evaluate(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(passed=True, reason="all checks passed")


class _CashThatMustNotReserve:
    reserved_cash = Decimal("0")

    def reserve_cash(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("sell approval must not reserve cash")


@pytest.mark.asyncio
async def test_sell_approval_does_not_create_cash_reservation() -> None:
    event_bus = InMemoryEventBus()
    controller = ApproveOrdersControllerImpl(
        gate=_PassingGate(),  # type: ignore[arg-type]
        cash_engine=_CashThatMustNotReserve(),  # type: ignore[arg-type]
        order_repo=InMemoryOrderRepository(),
        event_bus=event_bus,
        limits=SimpleNamespace(),  # type: ignore[arg-type]
    )
    intent = _sell_intent()

    approved, rejected = await controller.approve(
        [intent],
        _account_with_position(),
    )

    assert rejected == []
    assert len(approved) == 1
    assert approved[0].cash_reservation_id is None
    events = [event for event in event_bus.history if isinstance(event, OrderApproved)]
    assert len(events) == 1
    assert events[0].reservation_id is None
