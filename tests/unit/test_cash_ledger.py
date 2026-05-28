"""Unit tests for CashLedger.

Covers:
- Basic buy eligibility and cash gate
- Sell always passes
- Reservation lifecycle: create, release, idempotency
- Partial fill: reservation kept alive until is_order_complete=True
- Full fill: reservation released, settled cash deducted
- Fill idempotency: same fill_id applied twice is a no-op
- Sell fill: proceeds enter unsettled pool
- Settlement: settle_lot moves proceeds to settled
- Settlement idempotency: same lot settled twice is a no-op
- Premature settlement: raises PrematureSettlementError
- Stale reservation expiry
- cancel_order convenience method
- Submitted order reservation protected from TTL expiry
- Deterministic settlement lot identity from fill_id
- Settlement lot fill_id deduplication
- Settlement lot underflow protection
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.domain.settlement import ReservationStatus, SettlementLot, SettlementStatus
from quant_platform.core.exceptions import (
    DuplicateReservationError,
    PrematureSettlementError,
)
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import SettlementCalendar

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_UTC = UTC
# 2024-06-03 is a Monday (post T+1 effective date), good for settlement tests
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_TODAY = date(2024, 6, 3)
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

    def advance(self, days: int = 0, minutes: int = 0) -> None:
        self._now += timedelta(days=days, minutes=minutes)


def _account(
    settled: Decimal,
    reserved: Decimal = Decimal("0"),
    unsettled: Decimal = Decimal("0"),
    positions: tuple[PositionSnapshot, ...] = (),
) -> AccountSnapshot:
    available = settled - reserved
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=settled,
        unsettled_cash=unsettled,
        reserved_cash=reserved,
        available_cash=available,
        net_asset_value=settled + unsettled,
        positions=positions,
    )


def _position(
    instrument_id: uuid.UUID, qty: int, price: Decimal = Decimal("50")
) -> PositionSnapshot:
    return PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=qty,
        average_cost=price,
        market_price=price,
        market_value=Decimal(qty) * price,
        unrealised_pnl=Decimal("0"),
        as_of=_NOW,
    )


def _buy(qty: int, limit_price: Decimal | None = None) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=limit_price,
    )


def _sell(qty: int) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=_INSTRUMENT,
        side=OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )


def _fill(order_id: uuid.UUID, side: OrderSide, qty: int, price: Decimal) -> FillEvent:
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="IB-001",
        instrument_id=_INSTRUMENT,
        side=side,
        quantity=qty,
        fill_price=price,
        commission=Decimal("1"),
        currency="USD",
        executed_at=_NOW,
        received_at=_NOW,
    )


@pytest.fixture
def ledger() -> CashLedger:
    return CashLedger(
        clock=_FixedClock(),
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=_account(Decimal("10000")),
    )


# ---------------------------------------------------------------------------
# can_open_order
# ---------------------------------------------------------------------------


class TestCanOpenOrder:
    def test_buy_with_sufficient_cash_approved(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))  # 500 × 1.01 = 505
        decision = ledger.can_open_order(intent, _account(Decimal("10000")))
        assert decision.approved

    def test_buy_with_insufficient_cash_rejected(self, ledger: CashLedger) -> None:
        intent = _buy(250, limit_price=Decimal("50"))  # 12500 × 1.01 > 10000
        decision = ledger.can_open_order(intent, _account(Decimal("10000")))
        assert not decision.approved
        assert "insufficient" in decision.reason.lower()

    def test_sell_approved_with_sufficient_position(self, ledger: CashLedger) -> None:
        intent = _sell(50)
        account = _account(
            Decimal("0"),
            positions=(_position(_INSTRUMENT, qty=100),),
        )
        decision = ledger.can_open_order(intent, account)
        assert decision.approved

    def test_sell_rejected_without_position(self, ledger: CashLedger) -> None:
        intent = _sell(50)
        decision = ledger.can_open_order(intent, _account(Decimal("0")))
        assert not decision.approved
        assert "insufficient position" in decision.reason

    def test_sell_rejected_with_insufficient_position(self, ledger: CashLedger) -> None:
        intent = _sell(50)
        account = _account(
            Decimal("0"),
            positions=(_position(_INSTRUMENT, qty=30),),
        )
        decision = ledger.can_open_order(intent, account)
        assert not decision.approved
        assert "insufficient position" in decision.reason

    def test_no_price_rejected(self, ledger: CashLedger) -> None:
        # MARKET buy with no positions in snapshot → no price available
        intent = _buy(10)
        decision = ledger.can_open_order(intent, _account(Decimal("10000")))
        assert not decision.approved
        assert "no price" in decision.reason.lower()

    def test_available_cash_used_not_snapshot_cash(self) -> None:
        """Ledger's available_cash, not account.available_cash, governs the check."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("1000")),  # ledger has $1000
        )
        stale_account = _account(Decimal("10000"))  # stale snapshot shows $10000
        # Order for $5000 → should fail because ledger has only $1000
        intent = _buy(100, limit_price=Decimal("50"))
        decision = ledger.can_open_order(intent, stale_account)
        assert not decision.approved


# ---------------------------------------------------------------------------
# Reservation lifecycle
# ---------------------------------------------------------------------------


class TestReservations:
    def test_reservation_reduces_available_cash(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        before = ledger.available_cash
        reservation = ledger.reserve_cash(intent, account)
        assert reservation.status == ReservationStatus.ACTIVE
        assert ledger.available_cash < before
        assert ledger.reserved_cash == reservation.reserved_amount

    def test_duplicate_reservation_raises(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        ledger.reserve_cash(intent, account)
        with pytest.raises(DuplicateReservationError):
            ledger.reserve_cash(intent, account)

    def test_reserve_cash_rejects_sell_orders(self, ledger: CashLedger) -> None:
        intent = _sell(10)
        account = _account(
            Decimal("10000"),
            positions=(_position(_INSTRUMENT, qty=25),),
        )
        with pytest.raises(ValueError, match="buy orders"):
            ledger.reserve_cash(intent, account)

    def test_release_restores_available_cash(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        original = ledger.available_cash
        reservation = ledger.reserve_cash(intent, account)
        ledger.release_reservation(reservation.reservation_id, "test")
        assert ledger.available_cash == original
        assert ledger.reserved_cash == Decimal("0")

    def test_release_is_idempotent(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        reservation = ledger.reserve_cash(intent, account)
        ledger.release_reservation(reservation.reservation_id, "first")
        ledger.release_reservation(reservation.reservation_id, "second")  # no-op
        assert ledger.reserved_cash == Decimal("0")


# ---------------------------------------------------------------------------
# Partial fill handling
# ---------------------------------------------------------------------------


class TestPartialFills:
    def test_partial_fill_keeps_reservation_alive(self, ledger: CashLedger) -> None:
        """Reservation must NOT be released on a partial fill."""
        intent = _buy(100, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        reservation = ledger.reserve_cash(intent, account)

        # Partial fill of 50 shares
        fill = _fill(intent.order_id, OrderSide.BUY, 50, Decimal("50"))
        ledger.apply_fill(fill, is_order_complete=False)

        # Reservation must still be ACTIVE
        assert ledger.reserved_cash == reservation.reserved_amount
        # Settled cash reduced by actual fill cost (50 × $50 + $1 commission)
        assert ledger.settled_cash == Decimal("10000") - (50 * 50 + 1)

    def test_full_fill_releases_reservation(self, ledger: CashLedger) -> None:
        """Reservation is released when is_order_complete=True."""
        intent = _buy(100, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        reservation = ledger.reserve_cash(intent, account)

        fill = _fill(intent.order_id, OrderSide.BUY, 100, Decimal("50"))
        ledger.apply_fill(fill, is_order_complete=True)

        assert ledger.reserved_cash == Decimal("0")
        assert ledger.settled_cash == Decimal("10000") - (100 * 50 + 1)

    def test_two_partial_fills_then_complete(self, ledger: CashLedger) -> None:
        """Two partial fills followed by final fill releases reservation correctly."""
        intent = _buy(100, limit_price=Decimal("50"))
        account = _account(Decimal("10000"))
        reservation = ledger.reserve_cash(intent, account)

        fill1 = _fill(intent.order_id, OrderSide.BUY, 40, Decimal("50"))
        fill2 = _fill(intent.order_id, OrderSide.BUY, 40, Decimal("50"))
        fill3 = _fill(intent.order_id, OrderSide.BUY, 20, Decimal("50"))

        ledger.apply_fill(fill1, is_order_complete=False)
        ledger.apply_fill(fill2, is_order_complete=False)

        # After two partial fills reservation still active
        assert ledger.reserved_cash == reservation.reserved_amount

        ledger.apply_fill(fill3, is_order_complete=True)

        # After final fill reservation released
        assert ledger.reserved_cash == Decimal("0")
        expected_cost = (40 + 40 + 20) * 50 + 3  # 3 fills × $1 commission
        assert ledger.settled_cash == Decimal("10000") - expected_cost


# ---------------------------------------------------------------------------
# Fill idempotency
# ---------------------------------------------------------------------------


class TestFillIdempotency:
    def test_duplicate_buy_fill_is_noop(self, ledger: CashLedger) -> None:
        """Applying the same buy fill_id twice must not double-deduct settled cash."""
        intent = _buy(10, limit_price=Decimal("50"))
        ledger.reserve_cash(intent, _account(Decimal("10000")))

        fill = _fill(intent.order_id, OrderSide.BUY, 10, Decimal("50"))
        ledger.apply_fill(fill, is_order_complete=True)
        after_first = ledger.settled_cash

        ledger.apply_fill(fill, is_order_complete=True)  # duplicate
        assert ledger.settled_cash == after_first  # unchanged

    def test_duplicate_sell_fill_is_noop(self, ledger: CashLedger) -> None:
        """Applying the same sell fill_id twice must not double-add unsettled cash."""
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("50"))

        ledger.apply_fill(fill)
        after_first = ledger.unsettled_cash

        ledger.apply_fill(fill)  # duplicate
        assert ledger.unsettled_cash == after_first  # unchanged


# ---------------------------------------------------------------------------
# Sell proceeds enter unsettled pool
# ---------------------------------------------------------------------------


class TestSellProceeds:
    def test_sell_fill_increases_unsettled_cash(self, ledger: CashLedger) -> None:
        intent = _sell(20)
        fill = _fill(intent.order_id, OrderSide.SELL, 20, Decimal("100"))
        # 20 × $100 - $1 commission = $1999 net
        ledger.apply_fill(fill)
        assert ledger.unsettled_cash == Decimal("1999")
        assert ledger.settled_cash == Decimal("10000")  # settled unchanged

    def test_sell_proceeds_not_available_for_buys(self, ledger: CashLedger) -> None:
        """Unsettled proceeds must not increase available_cash for new buys."""
        intent = _sell(20)
        fill = _fill(intent.order_id, OrderSide.SELL, 20, Decimal("100"))
        ledger.apply_fill(fill)
        # available_cash should still be based on settled_cash only
        assert ledger.available_cash == ledger.settled_cash - ledger.reserved_cash


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


class TestSettlement:
    def test_settle_lot_moves_proceeds_to_settled(self) -> None:
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        # Simulate a sell fill on Monday 2024-06-03
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(fill)  # adds $999 to unsettled

        # Build a settlement lot for trade date 2024-06-03 (settles 2024-06-04 T+1)
        lots = ledger.project_settlement([fill])
        lot = lots[0]

        # Still trade date — should raise PrematureSettlementError
        with pytest.raises(PrematureSettlementError):
            ledger.settle_lot(lot)

        # Advance clock to settlement date (2024-06-04)
        clock.advance(days=1)
        ledger.settle_lot(lot)

        assert ledger.unsettled_cash == Decimal("0")
        assert ledger.settled_cash == Decimal("10000") + Decimal("999")

    def test_settle_lot_is_idempotent(self) -> None:
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(fill)
        lots = ledger.project_settlement([fill])
        lot = lots[0]

        clock.advance(days=1)  # settlement date arrived
        ledger.settle_lot(lot)
        settled_after_first = ledger.settled_cash

        ledger.settle_lot(lot)  # duplicate — must be no-op
        assert ledger.settled_cash == settled_after_first

    def test_premature_settlement_raises(self, ledger: CashLedger) -> None:
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(fill)
        lots = ledger.project_settlement([fill])
        with pytest.raises(PrematureSettlementError):
            ledger.settle_lot(lots[0])


# ---------------------------------------------------------------------------
# Stale reservation expiry
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_stale_reservation_expires(self) -> None:
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _buy(10, limit_price=Decimal("50"))
        reservation = ledger.reserve_cash(intent, _account(Decimal("10000")))
        assert ledger.reserved_cash > Decimal("0")

        # Advance clock past TTL
        clock.advance(minutes=31)
        expired = ledger.expire_stale_reservations()

        assert reservation.reservation_id in expired
        assert ledger.reserved_cash == Decimal("0")

    def test_submitted_order_reservation_survives_ttl(self) -> None:
        """A reservation for a submitted order must NOT expire via TTL."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _buy(10, limit_price=Decimal("50"))
        reservation = ledger.reserve_cash(intent, _account(Decimal("10000")))
        ledger.mark_order_submitted(intent.order_id)

        clock.advance(minutes=60)  # well past TTL
        expired = ledger.expire_stale_reservations()

        assert reservation.reservation_id not in expired
        assert ledger.reserved_cash == reservation.reserved_amount

    def test_submitted_then_cancelled_can_expire(self) -> None:
        """After cancel_order, the reservation is released immediately."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _buy(10, limit_price=Decimal("50"))
        ledger.reserve_cash(intent, _account(Decimal("10000")))
        ledger.mark_order_submitted(intent.order_id)

        ledger.cancel_order(intent.order_id, "cancelled by operator")
        assert ledger.reserved_cash == Decimal("0")


# ---------------------------------------------------------------------------
# cancel_order convenience
# ---------------------------------------------------------------------------


class TestCancelOrder:
    def test_cancel_order_releases_reservation(self, ledger: CashLedger) -> None:
        intent = _buy(10, limit_price=Decimal("50"))
        reservation = ledger.reserve_cash(intent, _account(Decimal("10000")))
        assert ledger.reserved_cash == reservation.reserved_amount

        ledger.cancel_order(intent.order_id, "broker rejected")
        assert ledger.reserved_cash == Decimal("0")

    def test_cancel_order_with_no_reservation_is_noop(self, ledger: CashLedger) -> None:
        # No reservation exists; should not raise
        ledger.cancel_order(uuid.uuid4(), "no such order")


# ---------------------------------------------------------------------------
# Deterministic settlement lot identity
# ---------------------------------------------------------------------------


class TestDeterministicSettlement:
    def test_same_fill_produces_same_lot_id(self) -> None:
        """project_settlement called twice for the same fill must yield the same lot_id."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(fill)

        lots_first = ledger.project_settlement([fill])
        lots_second = ledger.project_settlement([fill])

        assert lots_first[0].lot_id == lots_second[0].lot_id

    def test_different_fills_produce_different_lot_ids(self) -> None:
        """Two distinct fills must produce different lot_ids."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _sell(10)
        fill_a = _fill(intent.order_id, OrderSide.SELL, 5, Decimal("100"))
        fill_b = _fill(intent.order_id, OrderSide.SELL, 5, Decimal("100"))
        ledger.apply_fill(fill_a)
        ledger.apply_fill(fill_b)

        lots_a = ledger.project_settlement([fill_a])
        lots_b = ledger.project_settlement([fill_b])

        assert lots_a[0].lot_id != lots_b[0].lot_id

    def test_settle_lot_deduplicates_by_fill_id(self) -> None:
        """Settling two lots derived from the same fill_id must not move cash twice."""
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _sell(10)
        fill = _fill(intent.order_id, OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(fill)  # +999 unsettled

        lots = ledger.project_settlement([fill])
        lot = lots[0]

        clock.advance(days=1)
        ledger.settle_lot(lot)
        settled_after = ledger.settled_cash

        # Create a second lot with a different lot_id but the same fill_id
        fake_lot = SettlementLot(
            lot_id=uuid.uuid4(),
            fill_id=fill.fill_id,
            instrument_id=fill.instrument_id,
            trade_date=lot.trade_date,
            settlement_date=lot.settlement_date,
            gross_proceeds=lot.gross_proceeds,
            commission=lot.commission,
            net_proceeds=lot.net_proceeds,
            currency="USD",
            status=SettlementStatus.PENDING,
        )
        ledger.settle_lot(fake_lot)  # must be no-op
        assert ledger.settled_cash == settled_after


# ---------------------------------------------------------------------------
# Settlement underflow protection
# ---------------------------------------------------------------------------


class TestSettlementUnderflow:
    def test_settle_lot_rejects_when_unsettled_would_go_negative(self) -> None:
        """settle_lot must reject if net_proceeds exceeds unsettled_cash."""
        from quant_platform.core.exceptions import SettlementError

        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        # No sell fills → unsettled_cash = 0, but try to settle a lot
        bogus_lot = SettlementLot(
            lot_id=uuid.uuid4(),
            fill_id=uuid.uuid4(),
            instrument_id=_INSTRUMENT,
            trade_date=_TODAY,
            settlement_date=_TODAY,
            gross_proceeds=Decimal("1001"),
            commission=Decimal("1"),
            net_proceeds=Decimal("1000"),
            currency="USD",
            status=SettlementStatus.PENDING,
        )
        with pytest.raises(SettlementError, match="exceeds unsettled_sell_proceeds"):
            ledger.settle_lot(bogus_lot)


# ---------------------------------------------------------------------------
# Broker-resync reset path
# ---------------------------------------------------------------------------


class TestLedgerResetFromSnapshot:
    def test_reset_clears_reservations_and_reseeds_cash(self) -> None:
        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )

        # Create active reservation and move some cash into unsettled.
        intent = _buy(10, limit_price=Decimal("50"))
        ledger.reserve_cash(intent, _account(Decimal("10000")))
        sell_fill = _fill(uuid.uuid4(), OrderSide.SELL, 10, Decimal("100"))
        ledger.apply_fill(sell_fill)

        assert ledger.reserved_cash > Decimal("0")
        assert ledger.unsettled_cash > Decimal("0")

        broker_snapshot = _account(
            settled=Decimal("7777"),
            unsettled=Decimal("123"),
        )
        ledger.reset_from_snapshot(broker_snapshot)

        assert ledger.settled_cash == Decimal("7777")
        assert ledger.unsettled_cash == Decimal("123")
        assert ledger.reserved_cash == Decimal("0")


# ---------------------------------------------------------------------------
# F2 — Partial fill out-of-order does not release reservation early
# ---------------------------------------------------------------------------


class TestPartialFillOrderSafety:
    """Reservation must stay alive until the order is explicitly complete."""

    def test_partial_fill_does_not_release_reservation(self) -> None:
        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10_000")),
        )
        # Use limit_price so reserve_cash can estimate cash needed.
        intent = _buy(100, limit_price=Decimal("10"))
        snap = _account(Decimal("10_000"))
        ledger.reserve_cash(intent, snap)
        reserved_before = ledger.reserved_cash
        assert reserved_before > Decimal("0")

        # First partial fill — is_order_complete=False
        fill1 = _fill(intent.order_id, OrderSide.BUY, 50, Decimal("10"))
        ledger.apply_fill(fill1, is_order_complete=False)

        # Reservation must still exist; available_cash should still be
        # constrained by the reservation.
        assert ledger.reserved_cash == reserved_before

    def test_final_fill_releases_reservation(self) -> None:
        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10_000")),
        )
        intent = _buy(100, limit_price=Decimal("10"))
        snap = _account(Decimal("10_000"))
        ledger.reserve_cash(intent, snap)
        ledger.mark_order_submitted(intent.order_id)

        fill1 = _fill(intent.order_id, OrderSide.BUY, 50, Decimal("10"))
        ledger.apply_fill(fill1, is_order_complete=False)
        assert ledger.reserved_cash > Decimal("0")

        fill2 = _fill(intent.order_id, OrderSide.BUY, 50, Decimal("10"))
        ledger.apply_fill(fill2, is_order_complete=True)
        # After the final fill the reservation should be gone.
        assert ledger.reserved_cash == Decimal("0")

    def test_duplicate_fill_id_is_noop(self) -> None:
        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10_000")),
        )
        intent = _buy(10, limit_price=Decimal("10"))
        ledger.reserve_cash(intent, _account(Decimal("10_000")))
        fill = _fill(intent.order_id, OrderSide.BUY, 10, Decimal("10"))

        settled_before = ledger.settled_cash
        ledger.apply_fill(fill, is_order_complete=True)
        settled_after = ledger.settled_cash

        # Re-delivering the same fill_id must not change state again.
        ledger.apply_fill(fill, is_order_complete=True)
        assert ledger.settled_cash == settled_after
        # Verify fill actually changed state (not a double-no-op).
        assert settled_after < settled_before


# ---------------------------------------------------------------------------
# Stream 1 — Zero-price guard and GC threshold (production hardening)
# ---------------------------------------------------------------------------


class TestZeroPriceGuard:
    """The price <= 0 guard in can_open_order() is defense-in-depth.

    Domain objects (OrderIntent, PositionSnapshot) already validate prices > 0,
    so this guard protects against future callers that bypass those validators
    (e.g. via object.__setattr__ on frozen dataclasses or external adapters).
    Tests use MagicMock to inject zero/negative prices into the internal path.
    """

    def test_zero_price_raises_data_staleness_error(self) -> None:
        from unittest.mock import MagicMock

        from quant_platform.core.exceptions import DataStalenessError

        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        # Mock an intent with no limit_price so the code falls back to market_price.
        mock_intent = MagicMock()
        mock_intent.side = __import__(
            "quant_platform.core.domain.orders", fromlist=["OrderSide"]
        ).OrderSide.BUY
        mock_intent.limit_price = None
        mock_intent.instrument_id = _INSTRUMENT
        mock_intent.quantity = 10

        # Mock account with a position whose market_price is 0.
        mock_pos = MagicMock()
        mock_pos.instrument_id = _INSTRUMENT
        mock_pos.market_price = Decimal("0")

        mock_account = MagicMock()
        mock_account.positions = [mock_pos]

        with pytest.raises(DataStalenessError):
            ledger.can_open_order(mock_intent, mock_account)

    def test_negative_price_raises_data_staleness_error(self) -> None:
        from unittest.mock import MagicMock

        from quant_platform.core.exceptions import DataStalenessError

        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        mock_intent = MagicMock()
        mock_intent.side = __import__(
            "quant_platform.core.domain.orders", fromlist=["OrderSide"]
        ).OrderSide.BUY
        mock_intent.limit_price = Decimal("-5")
        mock_intent.instrument_id = _INSTRUMENT
        mock_intent.quantity = 10

        with pytest.raises(DataStalenessError):
            ledger.can_open_order(mock_intent, _account(Decimal("10000")))

    def test_positive_price_does_not_raise(self) -> None:
        ledger = CashLedger(
            clock=_FixedClock(),
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("10000")),
        )
        intent = _buy(1, limit_price=Decimal("100"))
        result = ledger.can_open_order(intent, _account(Decimal("10000")))
        assert result.approved is True


class TestLotIdGcThreshold:
    def test_settled_lot_ids_compacted_on_threshold(self) -> None:
        """_settled_lot_ids must be compacted when it exceeds _LOT_ID_GC_THRESHOLD."""
        from quant_platform.services.portfolio_service.cash_ledger import (
            _LOT_ID_GC_THRESHOLD,
        )

        clock = _FixedClock()
        ledger = CashLedger(
            clock=clock,
            settlement_calendar=SettlementCalendar(),
            initial_snapshot=_account(Decimal("0")),
        )

        # Directly populate the idempotency sets to just over the threshold.
        overflow = _LOT_ID_GC_THRESHOLD + 10
        ledger._settled_lot_ids = {uuid.uuid4() for _ in range(overflow)}
        ledger._settled_fill_ids = {uuid.uuid4() for _ in range(overflow)}
        # Give the ledger enough unsettled cash for a settlement.
        ledger._unsettled_sell_proceeds = Decimal("1000")
        ledger._settled_cash = Decimal("0")

        today = _TODAY
        lot = SettlementLot(
            lot_id=uuid.uuid4(),
            fill_id=uuid.uuid4(),
            instrument_id=_INSTRUMENT,
            trade_date=today,
            settlement_date=today,
            gross_proceeds=Decimal("100"),
            commission=Decimal("1"),
            net_proceeds=Decimal("99"),
            currency="USD",
            status=SettlementStatus.PENDING,
        )
        ledger.settle_lot(lot)

        # After GC, the sets must be smaller than the threshold.
        assert len(ledger._settled_lot_ids) <= _LOT_ID_GC_THRESHOLD
        assert len(ledger._settled_fill_ids) <= _LOT_ID_GC_THRESHOLD
        # GC counters must record what was removed.
        assert ledger._gc_settled_lot_count > 0
        assert ledger._gc_settled_fill_count > 0
