"""Compose a primary and one or more secondary :class:`BarFetcher` sources.

If the primary raises, each secondary is tried in order until one returns
data (IB outage path).  If the primary returns no rows for a given instrument
in ``[start, end]``, the secondaries are queried in order for those names
only, and results are merged with primary rows winning on
``(instrument_id, date)`` conflicts.

Pass a list to ``secondaries`` for multi-vendor failover (e.g. Tiingo then
Polygon).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import TypeAlias

import structlog

from quant_platform.core.domain.instruments import Instrument
from quant_platform.core.domain.market_data import MarketBar

log = structlog.get_logger(__name__)

BarFetcher: TypeAlias = Callable[
    [list[Instrument], date, date],
    Awaitable[list[MarketBar]],
]


def _bar_key(bar: MarketBar) -> tuple[object, date]:
    return (bar.instrument_id, bar.timestamp.date())


@dataclass
class _MergeState:
    by_key: dict[tuple[object, date], MarketBar] = field(default_factory=dict)

    def add_all(self, bars: list[MarketBar], *, prefer_existing: bool) -> None:
        for b in bars:
            k = _bar_key(b)
            if prefer_existing and k in self.by_key:
                continue
            self.by_key[k] = b

    def as_list(self) -> list[MarketBar]:
        return list(self.by_key.values())


def _has_any_bars_in_window(
    bars: list[MarketBar],
    instrument_id: object,
    start: date,
    end: date,
) -> bool:
    for b in bars:
        if b.instrument_id != instrument_id:
            continue
        d = b.timestamp.date()
        if start <= d <= end:
            return True
    return False


class FailoverBarFetcher:
    """Try ``primary``; fall back through ``secondaries`` for gaps or on total failure.

    Secondaries are tried in order — the first one that returns data wins.
    Primary rows always win on ``(instrument_id, date)`` conflicts during
    partial-gap merging.
    """

    def __init__(
        self,
        primary: BarFetcher,
        *,
        secondaries: list[BarFetcher],
        primary_name: str = "primary",
        secondary_names: list[str] | None = None,
    ) -> None:
        self._secondaries = list(secondaries)
        if not self._secondaries:
            raise ValueError("FailoverBarFetcher requires at least one secondary fetcher")
        self._secondary_names = list(secondary_names or []) or [
            f"secondary_{i}" for i in range(len(self._secondaries))
        ]
        self._primary = primary
        self._primary_name = primary_name

    async def __call__(
        self,
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        if not instruments or end < start:
            return []

        try:
            primary_bars = await self._primary(instruments, start, end)
        except Exception as exc:  # noqa: BLE001 — boundary: any broker/SDK failure
            log.error(
                "failover_bar_fetcher.primary_raised",
                source=self._primary_name,
                error=str(exc),
                exc_info=True,
            )
            return await self._fallback_all(instruments, start, end)

        missing = [
            inst
            for inst in instruments
            if not _has_any_bars_in_window(primary_bars, inst.instrument_id, start, end)
        ]
        if not missing:
            return primary_bars

        log.warning(
            "failover_bar_fetcher.partial_gap",
            source=self._primary_name,
            missing_instruments=len(missing),
        )
        fill = await self._fallback_partial(missing, start, end)
        if not fill:
            return primary_bars

        state = _MergeState()
        state.add_all(primary_bars, prefer_existing=True)
        state.add_all(fill, prefer_existing=True)
        merged = state.as_list()
        log.info(
            "failover_bar_fetcher.merged",
            primary_rows=len(primary_bars),
            secondary_rows=len(fill),
            merged_rows=len(merged),
        )
        return merged

    async def _fallback_all(
        self,
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        """Try each secondary in order; return the first successful result."""
        for fetcher, name in zip(self._secondaries, self._secondary_names, strict=False):
            try:
                out = await fetcher(instruments, start, end)
                log.info(
                    "failover_bar_fetcher.fallback_all",
                    source=name,
                    bars=len(out),
                )
                return out
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "failover_bar_fetcher.secondary_raised",
                    source=name,
                    error=str(exc),
                    exc_info=True,
                )
        log.error("failover_bar_fetcher.all_secondaries_failed", instruments=len(instruments))
        return []

    async def _fallback_partial(
        self,
        missing: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        """Try each secondary for missing instruments; return the first non-empty result."""
        for fetcher, name in zip(self._secondaries, self._secondary_names, strict=False):
            try:
                fill = await fetcher(missing, start, end)
                if fill:
                    return fill
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "failover_bar_fetcher.secondary_raised",
                    source=name,
                    error=str(exc),
                    exc_info=True,
                )
        return []
