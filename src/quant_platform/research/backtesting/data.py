"""Data preparation helpers for vectorized research backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.common import _BACKTEST_WARMUP_DAYS

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.market_data.bars import MarketBar

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PreparedBacktestData:
    all_bars: dict[uuid.UUID, list[MarketBar]]
    price_series: dict[datetime, dict[uuid.UUID, Decimal]]
    feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]]
    regime_index_series: dict[datetime, list[float]]
    inline_feature_count: int

    @property
    def covered_instruments(self) -> int:
        return len(self.all_bars)


async def prepare_vectorbt_backtest_data(
    *,
    session: Any,
    instrument_ids: list[uuid.UUID],
    contracts_file: str,
    start: datetime,
    end: datetime,
    now: datetime,
    bar_seconds: int,
    rebalance_timestamps: list[datetime],
    feature_set_version: str,
) -> PreparedBacktestData:
    all_bars = await load_backtest_bars(
        session=session,
        instrument_ids=instrument_ids,
        contracts_file=contracts_file,
        start=start,
        end=end,
        bar_seconds=bar_seconds,
    )
    update_liquidity_profiles(
        session=session,
        all_bars=all_bars,
        end=end,
        now=now,
    )
    feature_series, price_series, inline_feature_count = await build_feature_price_series(
        session=session,
        instrument_ids=instrument_ids,
        all_bars=all_bars,
        rebalance_timestamps=rebalance_timestamps,
        feature_set_version=feature_set_version,
    )
    if not price_series:
        raise OperatorUsageError("No price data mapped to any rebalance timestamp.")
    regime_index_series = build_regime_index_series(
        all_bars=all_bars,
        rebalance_timestamps=rebalance_timestamps,
    )
    log.info(
        "backtest.data_ready",
        rebalance_with_prices=len(price_series),
        rebalance_with_features=len(feature_series),
        inline_computed=inline_feature_count,
        from_repo=len(feature_series) - inline_feature_count,
    )
    return PreparedBacktestData(
        all_bars=all_bars,
        price_series=price_series,
        feature_series=feature_series,
        regime_index_series=regime_index_series,
        inline_feature_count=inline_feature_count,
    )


async def load_backtest_bars(
    *,
    session: Any,
    instrument_ids: list[uuid.UUID],
    contracts_file: str,
    start: datetime,
    end: datetime,
    bar_seconds: int,
) -> dict[uuid.UUID, list[MarketBar]]:
    bar_window_start = start - timedelta(days=_BACKTEST_WARMUP_DAYS)
    all_bars: dict[uuid.UUID, list[MarketBar]] = {}
    log.info("backtest.loading_bars", instruments=len(instrument_ids))
    for instrument_id in instrument_ids:
        bars = await session.bar_store.get_bars(instrument_id, bar_seconds, bar_window_start, end)
        if bars:
            all_bars[instrument_id] = sorted(bars, key=lambda bar: bar.timestamp)
    if not all_bars:
        raise OperatorUsageError(
            "No bar data found for any instrument.\n"
            "Fetch first with:\n"
            f"  python -m quant_platform backfill"
            f" --contracts-file {contracts_file}"
            f" --start {(start - timedelta(days=_BACKTEST_WARMUP_DAYS)).date()}"
            f" --end {end.date()}"
        )
    log.info(
        "backtest.bars_loaded",
        instruments_with_history=len(all_bars),
        instruments_requested=len(instrument_ids),
        bars_total=sum(len(v) for v in all_bars.values()),
    )
    return all_bars


def update_liquidity_profiles(
    *,
    session: Any,
    all_bars: dict[uuid.UUID, list[MarketBar]],
    end: datetime,
    now: datetime,
) -> None:
    from quant_platform.services.data_service.reference.universe_manager import LiquidityProfile

    cutoff = end - timedelta(days=30)
    cutoff = cutoff.replace(tzinfo=UTC) if cutoff.tzinfo is None else cutoff
    profiles = []
    for instrument_id, bars in all_bars.items():
        window = [bar for bar in bars if bar.timestamp >= cutoff and bar.volume > 0]
        if len(window) >= 5:
            adv = sum(bar.volume for bar in window) / len(window)
            adv_usd = sum(bar.volume * float(bar.close) for bar in window) / len(window)
            profiles.append(
                LiquidityProfile(
                    instrument_id=instrument_id,
                    adv_shares_20d=adv,
                    adv_usd_20d=adv_usd,
                    last_close=Decimal(str(bars[-1].close)),
                    computed_at=now,
                )
            )
    if profiles:
        session.universe_manager.update_liquidity(profiles)
        log.info("backtest.liquidity_profiles", instruments=len(profiles))


def build_regime_index_series(
    *,
    all_bars: dict[uuid.UUID, list[MarketBar]],
    rebalance_timestamps: list[datetime],
) -> dict[datetime, list[float]]:
    proxy_bars = all_bars[next(iter(all_bars))]
    return {
        ts: [float(bar.close) for bar in proxy_bars if bar.timestamp <= ts]
        for ts in rebalance_timestamps
    }


async def build_feature_price_series(
    *,
    session: Any,
    instrument_ids: list[uuid.UUID],
    all_bars: dict[uuid.UUID, list[MarketBar]],
    rebalance_timestamps: list[datetime],
    feature_set_version: str,
) -> tuple[
    dict[datetime, dict[uuid.UUID, dict[str, float]]],
    dict[datetime, dict[uuid.UUID, Decimal]],
    int,
]:
    from quant_platform.services.research_service.features.cross_section.cross_section import (
        build_feature_bundle,
    )

    feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]] = {}
    price_series: dict[datetime, dict[uuid.UUID, Decimal]] = {}
    inline_feature_count = 0

    log.info("backtest.computing_features", rebalance_dates=len(rebalance_timestamps))
    for ts in rebalance_timestamps:
        prices: dict[uuid.UUID, Decimal] = {}
        for instrument_id, bars in all_bars.items():
            before = [bar for bar in bars if bar.timestamp <= ts]
            if before:
                prices[instrument_id] = Decimal(str(before[-1].close))
        if prices:
            price_series[ts] = prices

        vectors = await session.feature_repo.get_vectors(instrument_ids, feature_set_version, ts)
        if vectors:
            feature_series[ts] = {
                vector.instrument_id: {key: float(value) for key, value in vector.features.items()}
                for vector in vectors
            }
            continue

        bar_closes: dict[uuid.UUID, list[float]] = {}
        for instrument_id, bars in all_bars.items():
            closes = [float(bar.close) for bar in bars if bar.timestamp <= ts]
            if len(closes) >= 22:
                bar_closes[instrument_id] = closes
        if bar_closes:
            bundle = build_feature_bundle(bar_closes)
            if bundle.alpha_features:
                feature_series[ts] = bundle.alpha_features
                inline_feature_count += 1
    return feature_series, price_series, inline_feature_count
