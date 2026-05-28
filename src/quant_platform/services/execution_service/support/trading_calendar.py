"""US-equities trading calendar used by the execution path market-hours gate.

Implements a minimal, dependency-free calendar:
- Regular trading hours: 09:30-16:00 America/New_York, Mon-Fri
- Fixed NYSE early-close / closed holiday list through 2030

This is intentionally conservative: when in doubt we treat the session as
closed.  Real deployments that need half-days or additional holidays can
replace ``DefaultTradingCalendar`` with an adapter that consults
``pandas_market_calendars`` — the ``TradingCalendar`` protocol is all the
execution-path gate consumes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Protocol


class TradingCalendar(Protocol):
    """Minimal surface consumed by ``OrderThrottle`` for market-hours gating."""

    def is_open(self, ts: datetime) -> bool: ...


try:
    from zoneinfo import ZoneInfo  # Python 3.9+

    _NY: tzinfo = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback for stripped runtimes
    _NY = timezone(timedelta(hours=-5))


_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)

# Fixed US equity holiday list (2024-2030).  Half-days (early closes) are
# listed in ``_EARLY_CLOSE_DATES`` and close at 13:00 local.
_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2024
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 3, 29),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 11, 28),
        date(2024, 12, 25),
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
        date(2027, 7, 5),
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),
    }
)

_EARLY_CLOSE_DATES: frozenset[date] = frozenset(
    {
        date(2024, 7, 3),
        date(2024, 11, 29),
        date(2024, 12, 24),
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
        date(2026, 11, 27),
        date(2026, 12, 24),
    }
)

_EARLY_CLOSE = time(13, 0)


class DefaultTradingCalendar:
    """Default US-equities calendar implementation.

    Does not touch a network service; all rules are encoded above.  The
    implementation is deliberately trivial so it can be used in CI and
    backtests without additional dependencies.
    """

    def is_open(self, ts: datetime) -> bool:
        """Return True iff ``ts`` falls within the regular-hours session
        for the prevailing trading day in ``America/New_York``.

        The input must be timezone-aware; naive datetimes are rejected
        because "is the market open" is an inherently tz-qualified
        question and silently coercing to UTC would be a foot-gun.
        """
        if ts.tzinfo is None:
            raise ValueError("TradingCalendar.is_open requires tz-aware datetime")
        local = ts.astimezone(_NY)
        today = local.date()
        if local.weekday() >= 5:
            return False
        if today in _HOLIDAYS:
            return False
        open_time = _RTH_OPEN
        close_time = _EARLY_CLOSE if today in _EARLY_CLOSE_DATES else _RTH_CLOSE
        return open_time <= local.time() < close_time


class AlwaysOpenCalendar:
    """Test/CI calendar that always reports the market as open.

    Use only in backtests and unit tests that explicitly want to bypass
    the market-hours gate.
    """

    def is_open(self, ts: datetime) -> bool:
        del ts
        return True
