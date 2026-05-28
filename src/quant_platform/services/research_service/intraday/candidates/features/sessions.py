"""Intraday bar sessionization and session-level metric calculations."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data import INTRADAY_BAR_SECONDS
from quant_platform.services.research_service.campaigns.screening.common import ensure_utc
from quant_platform.services.research_service.intraday.candidates.features.types import (
    IntradaySessionMetrics,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from quant_platform.core.domain.market_data import MarketBar


def _sessions_by_instrument(
    bars: Sequence[MarketBar],
) -> dict[uuid.UUID, tuple[IntradaySessionMetrics, ...]]:
    grouped: dict[tuple[uuid.UUID, date], list[MarketBar]] = defaultdict(list)
    for bar in bars:
        if bar.bar_seconds == INTRADAY_BAR_SECONDS and bar.is_complete:
            ts = ensure_utc(bar.timestamp)
            grouped[(bar.instrument_id, ts.date())].append(bar)
    by_instrument: dict[uuid.UUID, list[IntradaySessionMetrics]] = defaultdict(list)
    for (instrument_id, session_date), session_bars in grouped.items():
        metrics = _session_metrics(instrument_id, session_date, session_bars)
        if metrics is not None:
            by_instrument[instrument_id].append(metrics)
    return {
        instrument_id: tuple(sorted(metrics, key=lambda item: item.end_at))
        for instrument_id, metrics in by_instrument.items()
    }


def _latest_session_before(
    sessions: Sequence[IntradaySessionMetrics],
    as_of: datetime,
) -> IntradaySessionMetrics | None:
    if not sessions:
        return None
    end_times = [session.end_at for session in sessions]
    idx = bisect_left(end_times, ensure_utc(as_of)) - 1
    return sessions[idx] if idx >= 0 else None


def _sessions_before(
    sessions: Sequence[IntradaySessionMetrics],
    as_of: datetime,
) -> tuple[IntradaySessionMetrics, ...]:
    if not sessions:
        return ()
    end_times = [session.end_at for session in sessions]
    idx = bisect_left(end_times, ensure_utc(as_of))
    return tuple(sessions[:idx])


def _session_metrics(
    instrument_id: uuid.UUID,
    session_date: date,
    bars: Sequence[MarketBar],
) -> IntradaySessionMetrics | None:
    ordered = sorted(bars, key=lambda bar: ensure_utc(bar.timestamp))
    first, last = ordered[0], ordered[-1]
    open_price, close_price = _price(first.open), _price(last.close)
    high_price = max(_price(bar.high) for bar in ordered)
    low_price = min(_price(bar.low) for bar in ordered)
    if min(open_price, close_price, high_price, low_price) <= 0:
        return None
    start_at, end_at = ensure_utc(first.timestamp), ensure_utc(last.timestamp)
    opening_close = _price(_window_bars(ordered, end=start_at + timedelta(minutes=30))[-1].close)
    closing_open = _price(_window_bars(ordered, start=end_at - timedelta(minutes=30))[0].open)
    vwap = _session_vwap(ordered)
    total_volume = sum(max(int(bar.volume), 0) for bar in ordered)
    close_volume = sum(
        max(int(bar.volume), 0)
        for bar in _window_bars(ordered, start=end_at - timedelta(minutes=30))
    )
    return IntradaySessionMetrics(
        instrument_id=instrument_id,
        session_date=session_date,
        start_at=start_at,
        end_at=end_at,
        opening_drive=_log_return(opening_close, open_price),
        close_pressure=_log_return(close_price, closing_open),
        vwap_pressure=_log_return(close_price, vwap) if vwap > 0 else 0.0,
        intraday_volatility=_intraday_volatility(ordered),
        range_expansion=(high_price - low_price) / open_price,
        session_return=_log_return(close_price, open_price),
        volume_share=(close_volume / total_volume) if total_volume > 0 else 0.0,
    )


def _window_bars(
    bars: Sequence[MarketBar],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[MarketBar, ...]:
    window = tuple(
        bar
        for bar in bars
        if (start is None or ensure_utc(bar.timestamp) >= start)
        and (end is None or ensure_utc(bar.timestamp) < end)
    )
    return window or (bars[0] if start is None else bars[-1],)


def _session_vwap(bars: Sequence[MarketBar]) -> float:
    volume = sum(max(int(bar.volume), 0) for bar in bars)
    if volume <= 0:
        return sum(_price(bar.close) for bar in bars) / len(bars)
    return sum(_price(bar.vwap or bar.close) * max(int(bar.volume), 0) for bar in bars) / volume


def _intraday_volatility(bars: Sequence[MarketBar]) -> float:
    closes = [_price(bar.close) for bar in bars if _price(bar.close) > 0]
    returns = [_log_return(closes[idx], closes[idx - 1]) for idx in range(1, len(closes))]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))


def _log_return(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return math.log(numerator / denominator)


def _price(raw: object) -> float:
    return float(str(raw))
