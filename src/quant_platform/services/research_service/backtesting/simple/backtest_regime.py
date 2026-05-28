"""Shared market-regime wiring for research backtest engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.core.algorithms.portfolio_construction import (
    SimpleRegimeDetector,
)
from quant_platform.core.domain.signals import RegimeState
from quant_platform.core.regime import (
    MarketRegimeDetector,
    MarketStats,
    RegimeThresholds,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

BacktestRegimeDetector = MarketRegimeDetector | SimpleRegimeDetector


def build_backtest_regime_detector(
    settings: PlatformSettings,
    regime_detector: BacktestRegimeDetector | None,
    *,
    error_suffix: str = "explicitly to opt out of regime-parity enforcement.",
) -> BacktestRegimeDetector:
    """Return the governed regime detector for a backtest run."""
    require_market_regime = settings.backtest.require_market_regime
    if regime_detector is None:
        if require_market_regime:
            t = settings.regime.thresholds
            return MarketRegimeDetector(
                thresholds=RegimeThresholds(
                    crisis_vol=t.crisis_vol,
                    risk_off_vol=t.risk_off_vol,
                    low_vol=t.low_vol,
                    downtrend_z=t.downtrend_z,
                    uptrend_z=t.uptrend_z,
                    weak_breadth=t.weak_breadth,
                    strong_breadth=t.strong_breadth,
                ),
            )
        return SimpleRegimeDetector()

    if require_market_regime and isinstance(regime_detector, SimpleRegimeDetector):
        raise ValueError(
            "settings.backtest.require_market_regime=True rejects "
            "SimpleRegimeDetector in run_with_data; pass a "
            f"MarketRegimeDetector or set require_market_regime=False {error_suffix}"
        )
    return regime_detector


def compute_backtest_market_stats(
    *,
    settings: PlatformSettings,
    as_of: datetime,
    index_closes: list[float] | None,
    history_closes: Mapping[uuid.UUID, list[float]],
    fallback_to_price_proxy: bool,
    log_event: str,
) -> MarketStats | None:
    """Compute regime statistics from explicit index or price-history proxy."""
    proxy_closes = index_closes
    if proxy_closes is None and fallback_to_price_proxy and history_closes:
        proxy_key = min(history_closes.keys())
        proxy_closes = history_closes[proxy_key]

    if not proxy_closes:
        return None

    regime_cfg = settings.regime
    try:
        return MarketRegimeDetector.compute_stats(
            index_closes=proxy_closes,
            instrument_closes=dict(history_closes),
            as_of=as_of,
            trend_window=regime_cfg.trend_window,
            vol_window=regime_cfg.vol_window,
            breadth_window=regime_cfg.breadth_window,
        )
    except Exception as exc:
        log.warning(log_event, error=str(exc), as_of=as_of.isoformat())
        return None


async def detect_backtest_regime_state(
    *,
    detector: BacktestRegimeDetector,
    settings: PlatformSettings,
    as_of: datetime,
    index_closes: list[float] | None,
    history_closes: Mapping[uuid.UUID, list[float]],
    fallback_to_price_proxy: bool,
    log_event: str,
) -> RegimeState:
    """Update a market detector from history, then return the regime state."""
    if hasattr(detector, "update") and not isinstance(detector, SimpleRegimeDetector):
        stats = compute_backtest_market_stats(
            settings=settings,
            as_of=as_of,
            index_closes=index_closes,
            history_closes=history_closes,
            fallback_to_price_proxy=fallback_to_price_proxy,
            log_event=log_event,
        )
        if stats is not None:
            detector.update(stats)

        current_stats = getattr(detector, "_current_stats", None)
        if current_stats is not None:
            stats_for_classify = MarketStats(
                trend_z=float(getattr(current_stats, "trend_z", 0.0)),
                realized_vol=float(getattr(current_stats, "realized_vol", 0.0)),
                breadth=float(getattr(current_stats, "breadth", 0.0)),
                as_of=as_of,
            )
            return detector.classify(stats_for_classify)

    detected = await detector.detect(as_of)
    return _coerce_regime_state(detected)


def _coerce_regime_state(value: object) -> RegimeState:
    if isinstance(value, RegimeState):
        return value
    raise TypeError(f"regime detector returned unexpected state type: {type(value).__name__}")
