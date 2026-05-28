"""Sampling date helpers."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import exchange_calendars as xcals


def research_as_of_dates(
    start: datetime,
    end: datetime,
    *,
    date_policy: str = "nyse-sessions",
) -> tuple[datetime, ...]:
    """Return governed research ``as_of`` dates using the requested policy."""
    if date_policy == "calendar-days":
        return daily_as_of_dates(start, end)
    if date_policy != "nyse-sessions":
        raise ValueError(f"unsupported research date policy: {date_policy}")
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    if end_utc < start_utc:
        raise ValueError("end must be >= start")
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        start_utc.date().isoformat(),
        end_utc.date().isoformat(),
    )
    as_of_dates = (datetime.combine(session.date(), time.min, tzinfo=UTC) for session in sessions)
    return tuple(as_of for as_of in as_of_dates if start_utc <= as_of <= end_utc)


def daily_as_of_dates(start: datetime, end: datetime) -> tuple[datetime, ...]:
    """Return daily UTC timestamps in ``[start, end]``."""
    current = _ensure_utc(start)
    stop = _ensure_utc(end)
    if stop < current:
        raise ValueError("end must be >= start")
    dates: list[datetime] = []
    while current <= stop:
        dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["_ensure_utc", "daily_as_of_dates", "research_as_of_dates"]
