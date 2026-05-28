"""Runtime market-regime statistics helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
    MarketStats,
)

if TYPE_CHECKING:
    from quant_platform.application.runtime.state import Session

log = structlog.get_logger(__name__)


async def compute_market_stats_from_store(
    session: Session,
    as_of: datetime,
) -> MarketStats | None:
    """Read market proxy/universe bars and compute rule-based regime stats."""
    regime_settings = session.settings.regime
    if not regime_settings.enabled:
        return None

    proxy_raw = regime_settings.market_proxy_instrument_id.strip()
    if not proxy_raw:
        return None
    try:
        proxy_id = uuid.UUID(proxy_raw)
    except ValueError:
        log.warning(
            "regime.market_proxy_invalid_uuid",
            value=proxy_raw,
        )
        return None

    bar_seconds = regime_settings.bar_seconds
    start = as_of - timedelta(days=regime_settings.lookback_days)

    try:
        proxy_bars = await session.bar_store.get_bars(
            proxy_id,
            bar_seconds,
            start,
            as_of,
        )
    except Exception as exc:
        log.warning("regime.proxy_bars_fetch_failed", error=str(exc))
        return None

    if not proxy_bars:
        log.warning("regime.proxy_bars_empty", proxy_id=str(proxy_id))
        return None

    index_closes = [float(b.close) for b in proxy_bars]

    universe_closes: dict[uuid.UUID, list[float]] = {}
    for inst in session.contract_master.list_active():
        if inst.instrument_id == proxy_id:
            continue
        try:
            bars = await session.bar_store.get_bars(
                inst.instrument_id,
                bar_seconds,
                start,
                as_of,
            )
        except Exception as exc:
            log.debug(
                "regime.universe_bars_fetch_failed",
                instrument_id=str(inst.instrument_id),
                error=str(exc),
            )
            continue
        if bars:
            universe_closes[inst.instrument_id] = [float(b.close) for b in bars]

    try:
        return MarketRegimeDetector.compute_stats(
            index_closes=index_closes,
            instrument_closes=universe_closes,
            as_of=as_of,
            trend_window=regime_settings.trend_window,
            vol_window=regime_settings.vol_window,
            breadth_window=regime_settings.breadth_window,
        )
    except Exception as exc:
        log.warning("regime.compute_stats_failed", error=str(exc))
        return None
