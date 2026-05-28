"""Unit tests for SettlementCalendar — T+1 and T+2 settlement date projection."""

from __future__ import annotations

from datetime import date

import pytest

from quant_platform.services.portfolio_service.settlement_calendar import (
    _T1_EFFECTIVE,
    SettlementCalendar,
)


@pytest.fixture
def cal() -> SettlementCalendar:
    return SettlementCalendar()


class TestSettlementDays:
    def test_post_cutover_uses_t1(self, cal: SettlementCalendar) -> None:
        assert cal.settlement_days(_T1_EFFECTIVE) == 1

    def test_pre_cutover_uses_t2(self, cal: SettlementCalendar) -> None:
        assert cal.settlement_days(date(2024, 5, 27)) == 2

    def test_day_before_cutover_uses_t2(self, cal: SettlementCalendar) -> None:
        assert cal.settlement_days(date(2024, 5, 24)) == 2  # last trading day before cutover


class TestSettlementDate:
    # ----------- T+1 regime (post 2024-05-28) -----------

    def test_monday_t1_settles_tuesday(self, cal: SettlementCalendar) -> None:
        # 2024-06-03 Monday → 2024-06-04 Tuesday
        assert cal.settlement_date(date(2024, 6, 3)) == date(2024, 6, 4)

    def test_friday_t1_settles_monday(self, cal: SettlementCalendar) -> None:
        # 2024-06-07 Friday → 2024-06-10 Monday (skip weekend)
        assert cal.settlement_date(date(2024, 6, 7)) == date(2024, 6, 10)

    def test_thursday_before_holiday_t1(self, cal: SettlementCalendar) -> None:
        # 2024-07-03 Wednesday (day before 4th of July) → 2024-07-05 Friday
        # 4th July 2024 is Thursday, a NYSE holiday
        assert cal.settlement_date(date(2024, 7, 3)) == date(2024, 7, 5)

    # ----------- T+2 regime (pre 2024-05-28) -----------

    def test_monday_t2_settles_wednesday(self, cal: SettlementCalendar) -> None:
        # 2024-05-20 Monday → 2024-05-22 Wednesday (T+2)
        assert cal.settlement_date(date(2024, 5, 20)) == date(2024, 5, 22)

    def test_thursday_t2_settles_monday(self, cal: SettlementCalendar) -> None:
        # 2024-05-23 Thursday → 2024-05-28 Tuesday (skip Memorial Day 2024-05-27)
        # Actually: T+2 from 2024-05-23 → next 2 business days
        # 2024-05-24 (Fri) = 1st, 2024-05-28 (Tue, Memorial Day is Mon) = 2nd
        result = cal.settlement_date(date(2024, 5, 23))
        assert result == date(2024, 5, 28)

    def test_friday_t2_settles_tuesday(self, cal: SettlementCalendar) -> None:
        # 2024-05-17 Friday → 2024-05-21 Tuesday (T+2, skip weekend)
        assert cal.settlement_date(date(2024, 5, 17)) == date(2024, 5, 21)

    def test_historical_pre_2020(self, cal: SettlementCalendar) -> None:
        # 2020-01-02 Thursday → T+2 → 2020-01-06 Monday
        assert cal.settlement_date(date(2020, 1, 2)) == date(2020, 1, 6)

    # ----------- Error cases -----------

    def test_non_trading_day_raises(self, cal: SettlementCalendar) -> None:
        with pytest.raises(ValueError, match="not a NYSE business day"):
            cal.settlement_date(date(2024, 6, 1))  # Saturday

    def test_holiday_raises(self, cal: SettlementCalendar) -> None:
        with pytest.raises(ValueError, match="not a NYSE business day"):
            cal.settlement_date(date(2024, 7, 4))  # NYSE holiday


class TestIsSettled:
    def test_settled_on_settlement_date(self, cal: SettlementCalendar) -> None:
        trade = date(2024, 6, 3)
        settle = cal.settlement_date(trade)
        assert cal.is_settled(trade, settle)

    def test_not_settled_on_trade_date(self, cal: SettlementCalendar) -> None:
        trade = date(2024, 6, 3)
        assert not cal.is_settled(trade, trade)

    def test_settled_day_after_settlement(self, cal: SettlementCalendar) -> None:
        from datetime import timedelta

        trade = date(2024, 6, 3)
        settle = cal.settlement_date(trade)
        assert cal.is_settled(trade, settle + timedelta(days=5))
