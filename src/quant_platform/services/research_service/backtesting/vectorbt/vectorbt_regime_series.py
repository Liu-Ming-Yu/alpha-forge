"""Regime-state series helpers for VectorBT backtests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..simple.backtest_regime import (
    detect_backtest_regime_state,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.config import PlatformSettings
    from quant_platform.core.algorithms.portfolio_construction import (
        SimpleRegimeDetector,
    )
    from quant_platform.core.domain.signals import RegimeState
    from quant_platform.core.regime import MarketRegimeDetector


async def compute_vectorbt_regime_series(
    *,
    settings: PlatformSettings,
    rebalance_timestamps: list[datetime],
    price_series: dict[datetime, dict[uuid.UUID, Decimal]],
    regime_detector: MarketRegimeDetector | SimpleRegimeDetector,
    regime_index_series: dict[datetime, list[float]] | None,
) -> dict[datetime, RegimeState]:
    """Pre-compute regime state for each VectorBT rebalance timestamp."""
    result: dict[datetime, RegimeState] = {}
    history_closes: dict[uuid.UUID, list[float]] = {}

    for ts in rebalance_timestamps:
        prices_at_ts = price_series.get(ts, {})
        for instrument_id, price in prices_at_ts.items():
            history_closes.setdefault(instrument_id, []).append(float(price))

        index_closes = regime_index_series.get(ts) if regime_index_series is not None else None
        regime_state = await detect_backtest_regime_state(
            detector=regime_detector,
            settings=settings,
            as_of=ts,
            index_closes=index_closes,
            history_closes=history_closes,
            fallback_to_price_proxy=False,
            log_event="vectorbt.regime_stats_failed",
        )

        result[ts] = regime_state

    return result
