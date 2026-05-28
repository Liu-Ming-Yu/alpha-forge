"""Unit tests for PreTradeComplianceChecker (Stream 3)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.services.execution_service.orders.pretrade_compliance import (
    DayTradeCounter,
    PreTradeComplianceChecker,
)

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_RUN = uuid.uuid4()
_TARGET = uuid.uuid4()
_INSTRUMENT = uuid.uuid4()
_OTHER = uuid.uuid4()


def _account(
    nav: Decimal = Decimal("50000"), positions: tuple[PositionSnapshot, ...] = ()
) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=nav,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=nav,
        net_asset_value=nav,
        positions=positions,
    )


def _position(instrument_id: uuid.UUID, qty: int) -> PositionSnapshot:
    return PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=qty,
        average_cost=Decimal("50"),
        market_price=Decimal("50"),
        market_value=Decimal(qty * 50),
        unrealised_pnl=Decimal("0"),
        as_of=_NOW,
    )


def _buy(qty: int = 1, instrument_id: uuid.UUID | None = None) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=instrument_id or _INSTRUMENT,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )


def _sell(qty: int = 1, instrument_id: uuid.UUID | None = None) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN,
        portfolio_target_id=_TARGET,
        instrument_id=instrument_id or _INSTRUMENT,
        side=OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )


# ---------------------------------------------------------------------------
# HALT rule
# ---------------------------------------------------------------------------


class TestHaltRule:
    def test_halted_instrument_is_blocked(self) -> None:
        checker = PreTradeComplianceChecker(halted_instruments={_INSTRUMENT})
        violations = checker.check(_buy(), _account())
        assert any(v.rule == "HALT" and v.severity == "BLOCK" for v in violations)

    def test_non_halted_instrument_passes(self) -> None:
        checker = PreTradeComplianceChecker(halted_instruments={_OTHER})
        violations = checker.check(_buy(), _account())
        assert not any(v.rule == "HALT" for v in violations)

    def test_halt_applies_to_sell_too(self) -> None:
        pos = _position(_INSTRUMENT, 10)
        checker = PreTradeComplianceChecker(halted_instruments={_INSTRUMENT})
        violations = checker.check(_sell(5), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert any(v.rule == "HALT" and v.severity == "BLOCK" for v in violations)


# ---------------------------------------------------------------------------
# CASH_NO_SHORT rule
# ---------------------------------------------------------------------------


class TestCashNoShortRule:
    def test_sell_more_than_held_is_blocked(self) -> None:
        pos = _position(_INSTRUMENT, 5)
        checker = PreTradeComplianceChecker()
        violations = checker.check(_sell(10), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert any(v.rule == "CASH_NO_SHORT" and v.severity == "BLOCK" for v in violations)

    def test_sell_exactly_held_is_allowed(self) -> None:
        pos = _position(_INSTRUMENT, 10)
        checker = PreTradeComplianceChecker()
        violations = checker.check(_sell(10), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert not any(v.rule == "CASH_NO_SHORT" for v in violations)

    def test_sell_less_than_held_is_allowed(self) -> None:
        pos = _position(_INSTRUMENT, 20)
        checker = PreTradeComplianceChecker()
        violations = checker.check(_sell(5), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert not any(v.rule == "CASH_NO_SHORT" for v in violations)

    def test_sell_with_no_position_is_blocked(self) -> None:
        checker = PreTradeComplianceChecker()
        # No position provided → available_qty = 0, sell qty = 1 → short
        violations = checker.check(_sell(1), _account())
        assert any(v.rule == "CASH_NO_SHORT" and v.severity == "BLOCK" for v in violations)

    def test_buy_never_triggers_no_short(self) -> None:
        checker = PreTradeComplianceChecker()
        violations = checker.check(_buy(100), _account())
        assert not any(v.rule == "CASH_NO_SHORT" for v in violations)


# ---------------------------------------------------------------------------
# PDT_LIMIT rule
# ---------------------------------------------------------------------------


class TestPdtLimitRule:
    def test_warn_when_nav_below_25k_and_3_day_trades(self) -> None:
        counter = DayTradeCounter()
        counter.increment()
        counter.increment()
        counter.increment()
        checker = PreTradeComplianceChecker(
            pdt_enabled=True,
            day_trades_today=counter,
        )
        low_nav_account = _account(nav=Decimal("24999"))
        violations = checker.check(_buy(), low_nav_account)
        assert any(v.rule == "PDT_LIMIT" and v.severity == "WARN" for v in violations)

    def test_no_warn_when_nav_above_25k(self) -> None:
        counter = DayTradeCounter()
        for _ in range(5):
            counter.increment()
        checker = PreTradeComplianceChecker(pdt_enabled=True, day_trades_today=counter)
        high_nav_account = _account(nav=Decimal("25001"))
        violations = checker.check(_buy(), high_nav_account)
        assert not any(v.rule == "PDT_LIMIT" for v in violations)

    def test_no_warn_when_fewer_than_3_day_trades(self) -> None:
        counter = DayTradeCounter()
        counter.increment()
        counter.increment()
        checker = PreTradeComplianceChecker(pdt_enabled=True, day_trades_today=counter)
        violations = checker.check(_buy(), _account(nav=Decimal("10000")))
        assert not any(v.rule == "PDT_LIMIT" for v in violations)

    def test_pdt_disabled_suppresses_check(self) -> None:
        counter = DayTradeCounter()
        for _ in range(5):
            counter.increment()
        checker = PreTradeComplianceChecker(pdt_enabled=False, day_trades_today=counter)
        violations = checker.check(_buy(), _account(nav=Decimal("1000")))
        assert not any(v.rule == "PDT_LIMIT" for v in violations)

    def test_pdt_is_warn_not_block(self) -> None:
        counter = DayTradeCounter()
        for _ in range(3):
            counter.increment()
        checker = PreTradeComplianceChecker(pdt_enabled=True, day_trades_today=counter)
        violations = checker.check(_buy(), _account(nav=Decimal("10000")))
        pdt = next((v for v in violations if v.rule == "PDT_LIMIT"), None)
        assert pdt is not None
        assert pdt.severity == "WARN"

    def test_pdt_only_applies_to_buy_side(self) -> None:
        counter = DayTradeCounter()
        for _ in range(5):
            counter.increment()
        pos = _position(_INSTRUMENT, 10)
        checker = PreTradeComplianceChecker(pdt_enabled=True, day_trades_today=counter)
        violations = checker.check(
            _sell(5), _account(nav=Decimal("1000"), positions=(pos,)), {_INSTRUMENT: pos}
        )
        assert not any(v.rule == "PDT_LIMIT" for v in violations)


# ---------------------------------------------------------------------------
# WASH_SALE rule
# ---------------------------------------------------------------------------


class TestWashSaleRule:
    def test_warn_when_sold_within_lookback(self) -> None:
        # Use wall-clock time because _check_wash_sale calls datetime.now()
        from datetime import datetime as _dt

        sold_at = _dt.now(tz=_UTC) - timedelta(days=15)
        checker = PreTradeComplianceChecker(
            wash_sale_lookback_days=30,
            sell_history={_INSTRUMENT: sold_at},
        )
        violations = checker.check(_buy(), _account())
        assert any(v.rule == "WASH_SALE" and v.severity == "WARN" for v in violations)

    def test_no_warn_when_sold_outside_lookback(self) -> None:
        from datetime import datetime as _dt

        sold_at = _dt.now(tz=_UTC) - timedelta(days=31)
        checker = PreTradeComplianceChecker(
            wash_sale_lookback_days=30,
            sell_history={_INSTRUMENT: sold_at},
        )
        violations = checker.check(_buy(), _account())
        assert not any(v.rule == "WASH_SALE" for v in violations)

    def test_no_warn_when_no_sell_history(self) -> None:
        checker = PreTradeComplianceChecker(wash_sale_lookback_days=30)
        violations = checker.check(_buy(), _account())
        assert not any(v.rule == "WASH_SALE" for v in violations)

    def test_wash_sale_only_applies_to_buy_side(self) -> None:
        from datetime import datetime as _dt

        sold_at = _dt.now(tz=_UTC) - timedelta(days=5)
        pos = _position(_INSTRUMENT, 10)
        checker = PreTradeComplianceChecker(
            wash_sale_lookback_days=30,
            sell_history={_INSTRUMENT: sold_at},
        )
        violations = checker.check(_sell(5), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert not any(v.rule == "WASH_SALE" for v in violations)

    def test_record_sell_updates_history(self) -> None:
        from datetime import datetime as _dt

        checker = PreTradeComplianceChecker(wash_sale_lookback_days=30)
        assert not any(v.rule == "WASH_SALE" for v in checker.check(_buy(), _account()))
        checker.record_sell(_INSTRUMENT, _dt.now(tz=_UTC) - timedelta(days=10))
        violations = checker.check(_buy(), _account())
        assert any(v.rule == "WASH_SALE" for v in violations)


# ---------------------------------------------------------------------------
# Clean order (no violations)
# ---------------------------------------------------------------------------


class TestCleanOrder:
    def test_normal_buy_has_no_violations(self) -> None:
        checker = PreTradeComplianceChecker()
        violations = checker.check(_buy(), _account())
        assert violations == []

    def test_sell_within_holdings_has_no_violations(self) -> None:
        pos = _position(_INSTRUMENT, 10)
        checker = PreTradeComplianceChecker()
        violations = checker.check(_sell(5), _account(positions=(pos,)), {_INSTRUMENT: pos})
        assert violations == []


# ---------------------------------------------------------------------------
# DayTradeCounter
# ---------------------------------------------------------------------------


class TestDayTradeCounter:
    def test_initial_count_is_zero(self) -> None:
        assert DayTradeCounter().count == 0

    def test_increment_increases_count(self) -> None:
        counter = DayTradeCounter()
        counter.increment()
        counter.increment()
        assert counter.count == 2
