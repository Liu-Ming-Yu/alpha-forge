"""Application use case for durable historical feature-vector backfill."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from quant_platform.application.features.backfill_types import (
    FeatureBackfillDay,
    FeatureBackfillResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.core.contracts import FeatureRepository, HistoricalDataStore
    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar

_MIN_HISTORY_BARS = 21


class FeatureSetBackfiller(Protocol):
    async def __call__(
        self,
        *,
        bar_history: Mapping[uuid.UUID, Sequence[MarketBar]],
        feature_repo: FeatureRepository,
        strategy_run_id: uuid.UUID,
        as_of: datetime,
        feature_set_version: str,
        artifact_uri: str,
        dry_run: bool,
        events_by_instrument: Mapping[uuid.UUID, Sequence[datetime]] | None = None,
    ) -> int | None: ...


async def run_feature_backfill(
    *,
    instruments: Sequence[Instrument],
    bar_store: HistoricalDataStore,
    feature_repo: FeatureRepository,
    artifact_uri: str,
    start: datetime,
    end: datetime,
    feature_set_version: str,
    bar_seconds: int,
    lookback_days: int,
    dry_run: bool,
    date_policy: str = "nyse-sessions",
    events_by_instrument: Mapping[uuid.UUID, Sequence[datetime]] | None = None,
    feature_set_backfiller: FeatureSetBackfiller | None = None,
) -> FeatureBackfillResult:
    from quant_platform.services.research_service.sampling.samples import research_as_of_dates

    daily_rows: list[FeatureBackfillDay] = []
    as_of_dates = tuple(research_as_of_dates(start, end, date_policy=date_policy))
    bar_cache = (
        await _load_bar_history_cache(
            instruments=instruments,
            bar_store=bar_store,
            start=min(as_of_dates),
            end=max(as_of_dates),
            lookback_days=lookback_days,
            bar_seconds=bar_seconds,
        )
        if as_of_dates
        else {}
    )

    for as_of in as_of_dates:
        bar_history, skipped = _slice_bar_history(
            instruments=instruments,
            cached_bars=bar_cache,
            as_of=as_of,
            lookback_days=lookback_days,
        )
        feature_vectors, skipped_existing = await _backfill_day(
            bar_history=bar_history,
            feature_repo=feature_repo,
            strategy_run_id=uuid.uuid4(),
            as_of=as_of,
            feature_set_version=feature_set_version,
            artifact_uri=artifact_uri,
            events_by_instrument=events_by_instrument,
            dry_run=dry_run,
            feature_set_backfiller=feature_set_backfiller,
        )
        daily_rows.append(
            FeatureBackfillDay(
                as_of=as_of,
                instruments_requested=len(instruments),
                instruments_with_history=len(bar_history),
                feature_vectors=feature_vectors,
                skipped_insufficient_history=skipped,
                skipped_existing_vectors=skipped_existing,
            )
        )

    return FeatureBackfillResult(
        feature_set_version=feature_set_version,
        date_policy=date_policy,
        dry_run=dry_run,
        start=start,
        end=end,
        days=tuple(daily_rows),
    )


async def _load_bar_history_cache(
    *,
    instruments: Sequence[Instrument],
    bar_store: HistoricalDataStore,
    start: datetime,
    end: datetime,
    lookback_days: int,
    bar_seconds: int,
) -> dict[uuid.UUID, list[MarketBar]]:
    history_start = start - timedelta(days=lookback_days)
    cache: dict[uuid.UUID, list[MarketBar]] = {}
    for instrument in instruments:
        bars = await bar_store.get_bars(
            instrument.instrument_id,
            bar_seconds,
            history_start,
            end,
        )
        cache[instrument.instrument_id] = sorted(bars, key=lambda bar: bar.timestamp)
    return cache


def _slice_bar_history(
    *,
    instruments: Sequence[Instrument],
    cached_bars: Mapping[uuid.UUID, Sequence[MarketBar]],
    as_of: datetime,
    lookback_days: int,
) -> tuple[dict[uuid.UUID, list[MarketBar]], int]:
    start = as_of - timedelta(days=lookback_days)
    history: dict[uuid.UUID, list[MarketBar]] = {}
    skipped = 0
    for instrument in instruments:
        ordered = [
            bar
            for bar in cached_bars.get(instrument.instrument_id, ())
            if start <= bar.timestamp <= as_of
        ]
        if len(ordered) < _MIN_HISTORY_BARS:
            skipped += 1
            continue
        history[instrument.instrument_id] = ordered
    return history, skipped


async def _backfill_day(
    *,
    bar_history: Mapping[uuid.UUID, Sequence[MarketBar]],
    feature_repo: FeatureRepository,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    artifact_uri: str,
    events_by_instrument: Mapping[uuid.UUID, Sequence[datetime]] | None,
    dry_run: bool,
    feature_set_backfiller: FeatureSetBackfiller | None,
) -> tuple[int, int]:
    if not bar_history:
        return 0, 0

    missing_history, skipped_existing = await _filter_existing_vectors(
        bar_history=bar_history,
        feature_repo=feature_repo,
        feature_set_version=feature_set_version,
        as_of=as_of,
    )
    if not missing_history:
        return 0, skipped_existing
    if feature_set_backfiller is None:
        return 0, skipped_existing

    feature_count = await feature_set_backfiller(
        bar_history=missing_history,
        feature_repo=feature_repo,
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        feature_set_version=feature_set_version,
        artifact_uri=artifact_uri,
        dry_run=dry_run,
        events_by_instrument=events_by_instrument,
    )
    return (feature_count or 0), skipped_existing


async def _filter_existing_vectors(
    *,
    bar_history: Mapping[uuid.UUID, Sequence[MarketBar]],
    feature_repo: FeatureRepository,
    feature_set_version: str,
    as_of: datetime,
) -> tuple[dict[uuid.UUID, Sequence[MarketBar]], int]:
    existing = await feature_repo.get_vectors(
        list(bar_history),
        feature_set_version,
        as_of,
    )
    existing_current = {
        vector.instrument_id
        for vector in existing
        if vector.feature_set_version == feature_set_version and vector.as_of == as_of
    }
    if not existing_current:
        return dict(bar_history), 0
    return (
        {
            instrument_id: bars
            for instrument_id, bars in bar_history.items()
            if instrument_id not in existing_current
        },
        len(existing_current),
    )


__all__ = [
    "FeatureBackfillDay",
    "FeatureBackfillResult",
    "FeatureSetBackfiller",
    "run_feature_backfill",
]
