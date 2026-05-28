"""Tests for the ``buy_side_t1_settlement`` feature flag on ``CashLedger``."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from quant_platform.config import CashSettings
from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import (
    SettlementCalendar,
)

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_INSTRUMENT = uuid.uuid4()
_RUN = uuid.uuid4()
_TARGET = uuid.uuid4()


class _FixedClock:
    def __init__(self, now: datetime = _NOW) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def advance(self, *, days: int = 0) -> None:
        self._now += timedelta(days=days)


def _account(settled: Decimal) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=settled,
        net_asset_value=settled,
        positions=(),
    )


def _buy(qty: int, price: Decimal) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=price,
    )


def _fill(order_id: uuid.UUID, qty: int, price: Decimal) -> FillEvent:
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="IB-T1",
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        fill_price=price,
        commission=Decimal("1"),
        currency="USD",
        executed_at=_NOW,
        received_at=_NOW,
    )


def _make_ledger(
    *, flag: bool, starting: Decimal = Decimal("10000")
) -> tuple[CashLedger, _FixedClock]:
    clock = _FixedClock()
    ledger = CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=_account(starting),
        settings=CashSettings(buy_side_t1_settlement=flag),
    )
    return ledger, clock


def test_flag_off_preserves_instant_settled_debit() -> None:
    """With the flag off, buy fills debit settled_cash immediately and
    release the reservation — baseline behavior unchanged."""
    ledger, _ = _make_ledger(flag=False)
    intent = _buy(10, Decimal("50"))
    acct = _account(Decimal("10000"))

    reservation = ledger.reserve_cash(intent, acct)
    fill = _fill(intent.order_id, 10, Decimal("50"))
    ledger.apply_fill(fill, is_order_complete=True)

    assert ledger.settled_cash == Decimal("9499")  # 10000 - (500 + 1 commission)
    assert ledger.unsettled_cash == Decimal("0")
    assert ledger.is_reservation_active(reservation.reservation_id) is False


def test_flag_on_parks_cost_in_unsettled_and_keeps_reservation() -> None:
    """With the flag on, buy fills are mirrored in an internal buy-debit
    pool (not visible via unsettled_cash) and keep the reservation ACTIVE
    until settlement, so available_cash stays correctly held.
    unsettled_cash reflects sell proceeds only and must remain zero here."""
    ledger, _ = _make_ledger(flag=True)
    intent = _buy(10, Decimal("50"))
    acct = _account(Decimal("10000"))

    reservation = ledger.reserve_cash(intent, acct)
    before_available = ledger.available_cash
    fill = _fill(intent.order_id, 10, Decimal("50"))
    ledger.apply_fill(fill, is_order_complete=True)

    assert ledger.settled_cash == Decimal("10000")
    assert ledger.unsettled_cash == Decimal("0")  # sell proceeds pool, not buy debit
    assert ledger.is_reservation_active(reservation.reservation_id) is True
    assert ledger.available_cash == before_available


def test_flag_on_settles_on_settlement_date() -> None:
    """``settle_pending_buys(today)`` on T+1 moves the debit from
    unsettled_cash to settled_cash and releases the reservation."""
    ledger, clock = _make_ledger(flag=True)
    intent = _buy(10, Decimal("50"))
    acct = _account(Decimal("10000"))

    reservation = ledger.reserve_cash(intent, acct)
    fill = _fill(intent.order_id, 10, Decimal("50"))
    ledger.apply_fill(fill, is_order_complete=True)

    clock.advance(days=3)
    settled_count = ledger.settle_pending_buys(clock.today())

    assert settled_count == 1
    assert ledger.settled_cash == Decimal("9499")
    assert ledger.unsettled_cash == Decimal("0")
    assert ledger.is_reservation_active(reservation.reservation_id) is False
    assert ledger.settle_pending_buys(clock.today()) == 0


def test_flag_on_settle_skips_future_dated_buys() -> None:
    """``settle_pending_buys`` must not touch fills whose settlement date
    has not yet arrived."""
    ledger, clock = _make_ledger(flag=True)
    intent = _buy(10, Decimal("50"))
    ledger.reserve_cash(intent, _account(Decimal("10000")))
    ledger.apply_fill(_fill(intent.order_id, 10, Decimal("50")), is_order_complete=True)

    assert ledger.settle_pending_buys(clock.today()) == 0
    assert ledger.unsettled_cash == Decimal("0")  # buy debit is internal, not in sell pool


def test_flag_off_settle_is_noop() -> None:
    """When the flag is off, ``settle_pending_buys`` is always a no-op."""
    ledger, clock = _make_ledger(flag=False)
    intent = _buy(10, Decimal("50"))
    ledger.reserve_cash(intent, _account(Decimal("10000")))
    ledger.apply_fill(_fill(intent.order_id, 10, Decimal("50")), is_order_complete=True)

    assert ledger.settle_pending_buys(clock.today() + timedelta(days=30)) == 0


def test_flag_on_fill_idempotent() -> None:
    """Re-delivering the same buy fill must not double-count the
    internal buy-debit.  The sell-proceeds pool (unsettled_cash) stays 0."""
    ledger, _ = _make_ledger(flag=True)
    intent = _buy(10, Decimal("50"))
    ledger.reserve_cash(intent, _account(Decimal("10000")))
    fill = _fill(intent.order_id, 10, Decimal("50"))

    ledger.apply_fill(fill, is_order_complete=True)
    ledger.apply_fill(fill, is_order_complete=True)

    # settled_cash must not have changed twice (idempotency guard works)
    assert ledger.settled_cash == Decimal("10000")
    assert ledger.unsettled_cash == Decimal("0")
