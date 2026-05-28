"""Clock implementations for live and deterministic test use.

WallClock returns the real system time.  FakeClock allows tests and backtest
engines to control time progression explicitly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from quant_platform.core.contracts import Clock


class WallClock(Clock):
    """Returns the real system UTC time."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def today(self) -> date:
        return datetime.now(tz=UTC).date()


class FakeClock(Clock):
    """Deterministic clock for testing and simulation.

    Args:
        initial: Starting datetime (must be timezone-aware UTC).
    """

    def __init__(self, initial: datetime) -> None:
        if initial.tzinfo is None:
            raise ValueError("initial must be timezone-aware")
        self._now = initial

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def advance(self, **kwargs: float) -> None:
        """Advance the clock by the given timedelta arguments."""
        from datetime import timedelta

        self._now += timedelta(**kwargs)

    def set(self, dt: datetime) -> None:
        """Jump to a specific datetime."""
        if dt.tzinfo is None:
            raise ValueError("dt must be timezone-aware")
        self._now = dt
