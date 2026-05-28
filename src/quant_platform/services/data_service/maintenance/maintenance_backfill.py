"""Backfill helper for data maintenance supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.services.data_service.ingest.daily_ingest import (
    BarFetcher,
    IngestResult,
    run_daily_ingest,
)

if TYPE_CHECKING:
    from datetime import date

    from quant_platform.core.contracts import HistoricalDataStore
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.services.data_service.reference.universe_manager import UniverseManager

log = structlog.get_logger(__name__)


async def run_maintenance_backfill(
    *,
    instruments: list[Instrument] | None,
    bar_store: HistoricalDataStore | None,
    universe_manager: UniverseManager | None,
    bar_fetcher: BarFetcher | None,
    start: date,
    end: date,
) -> IngestResult:
    """One-shot bar-data backfill over ``[start, end]``."""
    if instruments is None or bar_store is None or universe_manager is None or bar_fetcher is None:
        raise ValueError(
            "backfill_once requires instruments, bar_store, "
            "universe_manager, and bar_fetcher to be supplied at "
            "construction."
        )
    if end < start:
        raise ValueError(f"backfill_once: end={end} must be >= start={start}")

    log.info(
        "maintenance_supervisor.backfill.start",
        start=str(start),
        end=str(end),
        instruments=len(instruments),
    )
    result = await run_daily_ingest(
        instruments=instruments,
        bar_store=bar_store,
        universe_manager=universe_manager,
        fetcher=bar_fetcher,
        trade_date=end,
        lookback_days=max(1, (end - start).days),
    )
    log.info(
        "maintenance_supervisor.backfill.complete",
        bars_fetched=result.bars_fetched,
        bars_stored=result.bars_stored,
        profiles=result.liquidity_profiles_updated,
        warnings=len(result.quality_warnings or []),
    )
    return result


__all__ = ["run_maintenance_backfill"]
