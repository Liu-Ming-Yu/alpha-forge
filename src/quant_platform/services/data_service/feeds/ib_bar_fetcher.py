"""IB Gateway adapter for the ``BarFetcher`` protocol consumed by
``run_daily_ingest``.

Wraps ``IBGatewayBrokerGateway.fetch_historical_bars`` so the daily-ingest
pipeline can backfill the Parquet bar store from Interactive Brokers.

The adapter fetches daily bars for each instrument in a single call per
instrument (IB returns the full range up to ``endDateTime`` inside the
configured duration).  Pacing and contract mapping are delegated to the
broker gateway; this module is a thin contract-mapping shim so the ingest
pipeline stays decoupled from the broker SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from datetime import date

    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar

log = structlog.get_logger(__name__)


class _HistoricalBarsBroker(Protocol):
    """Minimal broker surface required by ``IBBarFetcher``.

    Decouples the adapter from the concrete ``IBGatewayBrokerGateway`` class
    so unit tests can pass an in-memory fake without importing ``ibapi``.
    """

    async def fetch_historical_bars(
        self,
        instrument_id: object,
        bar_seconds: int,
        end_date: date,
        duration: str = ...,
        what_to_show: str = ...,
    ) -> list[MarketBar]: ...


class IBBarFetcher:
    """Adapter exposing a broker's ``fetch_historical_bars`` as a
    ``BarFetcher`` callable.

    The ``__call__`` signature matches the ``BarFetcher`` alias declared in
    ``daily_ingest`` (``instruments, start, end -> list[MarketBar]``).
    """

    def __init__(
        self,
        broker: _HistoricalBarsBroker,
        *,
        bar_seconds: int = 86400,
        what_to_show: str = "TRADES",
    ) -> None:
        self._broker = broker
        self._bar_seconds = bar_seconds
        self._what_to_show = what_to_show

    async def __call__(
        self,
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        """Fetch bars for ``instruments`` over ``[start, end]``.

        Uses a single ``reqHistoricalData`` per instrument with an IB
        duration string derived from the requested window length.  Any
        instrument that returns no bars is logged and skipped (the ingest
        pipeline reports it as a quality warning).
        """
        duration = _duration_string(start, end)
        all_bars: list[MarketBar] = []
        for inst in instruments:
            try:
                bars = await self._broker.fetch_historical_bars(
                    inst.instrument_id,
                    self._bar_seconds,
                    end,
                    duration=duration,
                    what_to_show=self._what_to_show,
                )
            except Exception as exc:
                log.warning(
                    "ib_bar_fetcher.instrument_failed",
                    symbol=inst.symbol,
                    error=str(exc),
                )
                continue

            windowed = [bar for bar in bars if start <= bar.timestamp.date() <= end]
            all_bars.extend(windowed)

        return all_bars


def _duration_string(start: date, end: date) -> str:
    """Convert a ``[start, end]`` date window into an IB duration string.

    IB's ``reqHistoricalData`` uses coarse duration tokens.  We pick the
    shortest token that still covers the requested window; daily ingest
    typically runs with ~30 day windows.
    """
    days = max(1, (end - start).days + 1)
    if days <= 30:
        return f"{days} D"
    weeks = (days + 6) // 7
    if weeks <= 52:
        return f"{weeks} W"
    months = (days + 29) // 30
    if months <= 12:
        return f"{months} M"
    years = (days + 364) // 365
    return f"{years} Y"
