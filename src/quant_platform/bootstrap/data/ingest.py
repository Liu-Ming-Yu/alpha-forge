"""Historical-bar ingest operation wiring (IB broker or market-data vendor)."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.bootstrap.session.public_api import create_paper_session

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import date

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.services.data_service.ingest.daily_ingest_types import IngestResult

log = structlog.get_logger(__name__)


async def ingest_bars(
    settings: PlatformSettings,
    *,
    start: date,
    end: date,
    instrument_contracts: Mapping[uuid.UUID, dict[str, object]],
    bar_seconds: int = 86400,
    source: str = "ib",
) -> UseCaseResult[None]:
    """Backfill the configured bar store from IB or a market-data vendor.

    ``source="ib"`` uses the broker historical-data API.  ``source="vendor"``
    uses the configured Tiingo/Polygon feed and skips the IB connection — this
    is required for large historical backfills where IB pacing limits make
    per-instrument fetches infeasible.
    """
    contracts = dict(instrument_contracts)
    if not contracts:
        raise OperatorUsageError("contracts-file contains no instruments.")
    if source not in ("ib", "vendor"):
        raise OperatorUsageError(f"Unknown ingest source {source!r}; expected 'ib' or 'vendor'.")

    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
    )
    instruments = session.contract_master.list_active()
    if not instruments:
        raise OperatorUsageError("No active instruments in contract master.")

    if source == "vendor":
        result = await _ingest_via_vendor(settings, session, instruments, start, end, bar_seconds)
    else:
        result = await _ingest_via_ib(
            settings, session, contracts, instruments, start, end, bar_seconds
        )

    log.info(
        "ingest.complete",
        source=source,
        bars_fetched=result.bars_fetched,
        bars_stored=result.bars_stored,
        profiles=result.liquidity_profiles_updated,
        warnings=len(result.quality_warnings or []),
        start=str(start),
        end=str(end),
        universe=len(instruments),
    )
    return UseCaseResult(
        message=(
            f"Ingest complete ({source}): {result.bars_fetched} bars fetched, "
            f"{result.bars_stored} stored across {len(instruments)} instruments."
        ),
        presentation=ResultPresentation.TEXT,
    )


async def _ingest_via_vendor(
    settings: PlatformSettings,
    session: object,
    instruments: list[Instrument],
    start: date,
    end: date,
    bar_seconds: int,
) -> IngestResult:
    """Backfill bars from the configured Tiingo/Polygon vendor (no IB)."""
    from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (
        build_vendor_bar_fetcher,
    )
    from quant_platform.services.data_service.ingest.daily_ingest import run_daily_ingest

    fetcher = build_vendor_bar_fetcher(settings, bar_seconds=bar_seconds)
    if fetcher is None:
        raise OperatorUsageError(
            "Vendor ingest requires a configured data vendor. Set "
            "QP__DATA_INGEST__BAR_FETCH_FALLBACK (or _CHAIN) and the vendor API key."
        )
    return await run_daily_ingest(
        instruments=instruments,
        bar_store=session.bar_store,  # type: ignore[attr-defined]
        universe_manager=session.universe_manager,  # type: ignore[attr-defined]
        fetcher=fetcher,
        trade_date=end,
        lookback_days=max(1, (end - start).days),
    )


async def _ingest_via_ib(
    settings: PlatformSettings,
    session: object,
    contracts: dict[uuid.UUID, dict[str, object]],
    instruments: list[Instrument],
    start: date,
    end: date,
    bar_seconds: int,
) -> IngestResult:
    """Backfill bars via the IB Gateway historical-data API."""
    from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (
        build_ingest_bar_fetcher,
    )
    from quant_platform.services.data_service.ingest.daily_ingest import run_daily_ingest
    from quant_platform.services.execution_service.gateways.broker_gateway import (
        IBGatewayBrokerGateway,
    )
    from quant_platform.services.execution_service.stores.pacing_store import build_pacing_store

    pacing_store = build_pacing_store(
        redis_url=settings.storage.redis_url or None,
        client_id=settings.broker.client_id,
        window_seconds=settings.broker.historical_bar_pacing_window_seconds,
    )
    broker = IBGatewayBrokerGateway(
        settings=settings.broker,
        instrument_contracts=contracts,
        pacing_store=pacing_store,
    )

    await broker.connect()
    try:
        fetcher = build_ingest_bar_fetcher(settings, broker, bar_seconds=bar_seconds)
        if fetcher is None:
            raise OperatorUsageError("IBGatewayBrokerGateway must support fetch_historical_bars")
        return await run_daily_ingest(
            instruments=instruments,
            bar_store=session.bar_store,  # type: ignore[attr-defined]
            universe_manager=session.universe_manager,  # type: ignore[attr-defined]
            fetcher=fetcher,
            trade_date=end,
            lookback_days=max(1, (end - start).days),
        )
    finally:
        await broker.disconnect()


__all__ = ["ingest_bars"]
