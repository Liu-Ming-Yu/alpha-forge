"""Shared DTOs and callback types for daily ingest."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date

from quant_platform.core.domain.instruments import Instrument
from quant_platform.core.domain.market_data import MarketBar

BarFetcher = Callable[
    [list[Instrument], date, date],
    Awaitable[list[MarketBar]],
]


@dataclass
class IngestResult:
    """Summary of one daily ingest run."""

    bars_fetched: int = 0
    bars_stored: int = 0
    instruments_processed: int = 0
    liquidity_profiles_updated: int = 0
    quality_warnings: list[str] | None = None
    bars_dropped: int = 0
    drops_by_symbol: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.quality_warnings is None:
            self.quality_warnings = []
        if self.drops_by_symbol is None:
            self.drops_by_symbol = {}
