"""Settlement calendar for US equities.

This module is the single source of truth for projecting settlement dates and
for determining whether proceeds have settled.

Settlement regime history:
- Before 2024-05-28: T+2 (two business days after trade date).
- 2024-05-28 onward: T+1 (one business day after trade date).

Both regimes are supported so that backtests using historical data produce
correct settlement dates without operator intervention.

Invariants:
- settlement_date(trade_date) always returns a NYSE business day.
- Saturday, Sunday, and NYSE holidays are never settlement dates.
- The exchange calendar is loaded once and cached.
- trade_date must be a NYSE trading day; ValueError is raised otherwise.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import exchange_calendars as xcals

# US equity T+1 settlement took effect on 28 May 2024.
_T1_EFFECTIVE = date(2024, 5, 28)

_T1_DAYS = 1  # business days to settlement after 2024-05-28
_T2_DAYS = 2  # business days to settlement before 2024-05-28


@lru_cache(maxsize=1)
def _get_nyse_calendar() -> xcals.ExchangeCalendar:
    """Return the NYSE exchange calendar.  Cached after first call."""
    return xcals.get_calendar("XNYS")


class SettlementCalendar:
    """US equity settlement calendar supporting both T+1 and T+2 regimes.

    Uses T+2 for trade dates before 2024-05-28 and T+1 for all later dates.
    This ensures backtests over historical data produce accurate settlement
    projections without requiring any special configuration.

    Must never:
        Return a settlement date that falls on a weekend or NYSE holiday.
        Apply T+1 logic for any trade date before 2024-05-28.
        Be bypassed; all settlement date projection must use this class.
    """

    def __init__(self) -> None:
        self._cal = _get_nyse_calendar()

    def is_business_day(self, d: date) -> bool:
        """Return True if d is a NYSE trading day (weekday and not a holiday).

        Args:
            d: The date to check.
        """
        return bool(self._cal.is_session(d.isoformat()))

    def settlement_days(self, trade_date: date) -> int:
        """Return the number of business days to settlement for trade_date.

        Args:
            trade_date: The trade execution date.

        Returns:
            1 for dates on or after 2024-05-28; 2 for earlier dates.
        """
        return _T1_DAYS if trade_date >= _T1_EFFECTIVE else _T2_DAYS

    def convention_for(self, trade_date: date) -> str:
        """Return the human-readable settlement convention (e.g. ``"T+1"``).

        Use at session start to log/announce the convention this ledger will
        apply, so operators can verify the broker is configured to match.
        US equities transitioned T+2 → T+1 on 2024-05-28; this method
        encodes that switch and is the canonical source of truth.
        """
        return f"T+{self.settlement_days(trade_date)}"

    def settlement_date(self, trade_date: date) -> date:
        """Return the expected settlement date for a trade executed on trade_date.

        Applies T+1 for dates on or after 2024-05-28, T+2 for earlier dates.

        Args:
            trade_date: The date on which the trade was executed.  Must be a
                NYSE trading day; raises ValueError otherwise.

        Returns:
            The first NYSE business day at least settlement_days(trade_date)
            business days after trade_date.

        Failure semantics:
            Raises ValueError if trade_date is not a NYSE business day.
        """
        if not self.is_business_day(trade_date):
            raise ValueError(
                f"{trade_date} is not a NYSE business day; cannot determine settlement"
            )
        days_forward = self.settlement_days(trade_date)
        candidate = trade_date + timedelta(days=1)
        business_days_found = 0
        while business_days_found < days_forward:
            if self.is_business_day(candidate):
                business_days_found += 1
            if business_days_found < days_forward:
                candidate += timedelta(days=1)
        return candidate

    def days_until_settlement(self, trade_date: date, as_of: date) -> int:
        """Return (settlement_date(trade_date) - as_of).days.

        Negative means the settlement date has already passed.

        Args:
            trade_date: The trade execution date.
            as_of: The reference date (typically today).
        """
        return (self.settlement_date(trade_date) - as_of).days

    def is_settled(self, trade_date: date, as_of: date) -> bool:
        """Return True if proceeds from trade_date have settled by as_of.

        Args:
            trade_date: Trade execution date.
            as_of: Reference date (typically today).
        """
        return self.settlement_date(trade_date) <= as_of
