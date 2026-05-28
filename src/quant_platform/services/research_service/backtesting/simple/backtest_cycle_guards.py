"""Per-cycle safety and regime helpers for backtest replay."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.algorithms.portfolio_construction import (
    SimpleRegimeDetector,
)
from quant_platform.core.exceptions import LookAheadBiasError

from .backtest_regime import (
    BacktestRegimeDetector,
    compute_backtest_market_stats,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.config import PlatformSettings


def assert_features_point_in_time(
    *,
    signal_time: datetime,
    feature_available_at: dict[datetime, datetime] | None,
) -> None:
    """Fail when features were not available by the signal timestamp."""
    if feature_available_at is None:
        return
    available_at = feature_available_at.get(signal_time)
    if available_at is None or available_at <= signal_time:
        return
    raise LookAheadBiasError(
        f"look-ahead bias detected at signal_date={signal_time.isoformat()}: "
        f"features became available at {available_at.isoformat()} which is "
        "after the signal date"
    )


def refresh_backtest_regime_detector(
    *,
    detector: BacktestRegimeDetector,
    settings: PlatformSettings,
    as_of: datetime,
    price_series_at_ts: dict[uuid.UUID, Decimal],
    history_closes: dict[uuid.UUID, list[float]],
    regime_index_series: dict[datetime, list[float]] | None,
) -> None:
    """Update a market-regime detector from the latest replay price history."""
    for instrument_id, price in price_series_at_ts.items():
        history_closes.setdefault(instrument_id, []).append(float(price))

    if not hasattr(detector, "update") or isinstance(detector, SimpleRegimeDetector):
        return

    stats = compute_backtest_market_stats(
        settings=settings,
        as_of=as_of,
        index_closes=(regime_index_series.get(as_of) if regime_index_series is not None else None),
        history_closes=history_closes,
        fallback_to_price_proxy=True,
        log_event="backtest.regime_stats_failed",
    )
    if stats is not None:
        detector.update(stats)
