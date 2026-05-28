"""Deterministic replay clock for research simulations."""

from __future__ import annotations

from datetime import date, datetime, timedelta


class FakeClock:
    """Deterministic clock for testing and simulation replay."""

    def __init__(self, initial: datetime) -> None:
        if initial.tzinfo is None:
            raise ValueError("initial must be timezone-aware")
        self._now = initial

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def advance(self, **kwargs: float) -> None:
        self._now += timedelta(**kwargs)

    def set(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            raise ValueError("dt must be timezone-aware")
        self._now = dt
