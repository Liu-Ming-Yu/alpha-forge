"""Tests for the market-hours gate in ``OrderThrottle``."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.execution_service.support.trading_calendar import (
    AlwaysOpenCalendar,
    DefaultTradingCalendar,
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
        created_at=datetime(2026, 1, 5, 15, 0, tzinfo=UTC),
        limit_price=Decimal("100"),
    )


def test_default_calendar_open_during_rth() -> None:
    cal = DefaultTradingCalendar()
    # 2026-04-23 is a regular trading Thursday.  15:00 UTC = 11:00 NY.
    assert cal.is_open(datetime(2026, 4, 23, 15, 0, tzinfo=UTC)) is True


def test_default_calendar_closed_outside_rth() -> None:
    cal = DefaultTradingCalendar()
    # 09:00 NY is before the open.
    assert cal.is_open(datetime(2026, 4, 23, 13, 0, tzinfo=UTC)) is False


def test_default_calendar_closed_on_weekend() -> None:
    cal = DefaultTradingCalendar()
    # Saturday 2026-04-25.
    assert cal.is_open(datetime(2026, 4, 25, 15, 0, tzinfo=UTC)) is False


def test_default_calendar_closed_on_holiday() -> None:
    cal = DefaultTradingCalendar()
    # 2026-12-25 is Christmas.
    assert cal.is_open(datetime(2026, 12, 25, 15, 0, tzinfo=UTC)) is False


def test_default_calendar_rejects_naive_datetime() -> None:
    cal = DefaultTradingCalendar()
    with pytest.raises(ValueError, match="tz-aware"):
        cal.is_open(datetime(2026, 4, 23, 15, 0))


def test_throttle_rejects_when_market_closed() -> None:
    clock = FakeClock(initial=datetime(2026, 4, 25, 15, 0, tzinfo=UTC))
    throttle = OrderThrottle(
        clock,
        trading_calendar=DefaultTradingCalendar(),
        trading_hours_enforced=True,
    )
    decision = throttle.can_submit(_intent())
    assert decision.approved is False
    assert decision.reason == "market_closed"


def test_throttle_approves_when_market_open() -> None:
    clock = FakeClock(initial=datetime(2026, 4, 23, 15, 0, tzinfo=UTC))
    throttle = OrderThrottle(
        clock,
        trading_calendar=DefaultTradingCalendar(),
        trading_hours_enforced=True,
    )
    decision = throttle.can_submit(_intent())
    assert decision.approved is True


def test_throttle_skips_calendar_when_enforcement_off() -> None:
    clock = FakeClock(initial=datetime(2026, 4, 25, 15, 0, tzinfo=UTC))
    throttle = OrderThrottle(
        clock,
        trading_calendar=DefaultTradingCalendar(),
        trading_hours_enforced=False,
    )
    assert throttle.can_submit(_intent()).approved is True


def test_throttle_requires_calendar_when_enforcement_on() -> None:
    clock = FakeClock(initial=datetime(2026, 4, 25, 15, 0, tzinfo=UTC))
    with pytest.raises(ValueError, match="trading_calendar must be supplied"):
        OrderThrottle(clock, trading_hours_enforced=True)


def test_always_open_calendar_for_tests() -> None:
    cal = AlwaysOpenCalendar()
    assert cal.is_open(datetime(2026, 12, 25, 3, 0, tzinfo=UTC)) is True
