"""Daily bar ingest orchestrator."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.data_service.ingest.daily_ingest_quality import (
    check_continuity as _check_continuity,
)
from quant_platform.services.data_service.ingest.daily_ingest_quality import (
    check_quality as _check_quality,
)
from quant_platform.services.data_service.ingest.daily_ingest_quality import (
    filter_invariant_violations as _filter_invariant_violations,
)
from quant_platform.services.data_service.ingest.daily_ingest_types import BarFetcher, IngestResult
from quant_platform.services.data_service.ingest.daily_liquidity import (
    compute_liquidity_profiles as _compute_liquidity_profiles,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import HistoricalDataStore
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.data_service.reference.universe_manager import UniverseManager

log = structlog.get_logger(__name__)

_CALENDAR_BUFFER_DAYS = 7

__all__ = ["BarFetcher", "IngestResult", "run_daily_ingest"]


async def run_daily_ingest(
    instruments: list[Instrument],
    bar_store: HistoricalDataStore,
    universe_manager: UniverseManager,
    fetcher: BarFetcher,
    trade_date: date,
    lookback_days: int = 21,
) -> IngestResult:
    """Execute the daily bar ingest pipeline."""
    result = IngestResult()
    now = datetime.now(tz=UTC)
    start_date = date.fromordinal(trade_date.toordinal() - lookback_days - _CALENDAR_BUFFER_DAYS)

    log.info(
        "daily_ingest.start",
        instruments=len(instruments),
        trade_date=str(trade_date),
    )

    raw_bars = await fetcher(instruments, start_date, trade_date)
    result.bars_fetched = len(raw_bars)

    symbol_by_id = {inst.instrument_id: inst.symbol for inst in instruments}
    bars, drops_by_symbol = _filter_invariant_violations(raw_bars, symbol_by_id)
    result.bars_dropped = sum(drops_by_symbol.values())
    result.drops_by_symbol = drops_by_symbol

    warnings = _check_quality(bars, instruments, trade_date)
    warnings += _check_continuity(bars, instruments)
    if drops_by_symbol:
        warnings.append(
            "ohlc_invariant_drops: "
            + ", ".join(f"{k}={v}" for k, v in sorted(drops_by_symbol.items()))
        )
    result.quality_warnings = warnings

    await bar_store.store_bars(bars)
    result.bars_stored = len(bars)

    profiles = _compute_liquidity_profiles(bars, instruments, trade_date, now)
    universe_manager.update_liquidity(profiles)
    result.liquidity_profiles_updated = len(profiles)
    result.instruments_processed = len(instruments)

    log.info(
        "daily_ingest.complete",
        bars_fetched=result.bars_fetched,
        bars_stored=result.bars_stored,
        bars_dropped=result.bars_dropped,
        profiles=result.liquidity_profiles_updated,
        warnings=len(warnings),
    )

    return result


async def refresh_liquidity_from_store(
    instruments: list[Instrument],
    bar_store: HistoricalDataStore,
    universe_manager: UniverseManager,
    as_of: datetime,
    lookback_days: int = 21,
) -> int:
    """Recompute ADV profiles from bars already stored in Parquet."""
    start = as_of - timedelta(days=lookback_days + 5)
    bars: list[MarketBar] = []
    for inst in instruments:
        inst_bars = await bar_store.get_bars(
            inst.instrument_id,
            86400,
            start,
            as_of,
        )
        bars.extend(inst_bars)

    profiles = _compute_liquidity_profiles(
        bars=bars,
        instruments=instruments,
        trade_date=as_of.date(),
        now=as_of,
    )
    universe_manager.update_liquidity(profiles)
    log.info(
        "daily_ingest.refresh_liquidity_from_store",
        instruments=len(instruments),
        profiles=len(profiles),
        as_of=str(as_of),
    )
    return len(profiles)
