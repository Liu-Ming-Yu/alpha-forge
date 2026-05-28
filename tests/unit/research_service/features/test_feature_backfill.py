"""Tests for the durable feature-vector backfill orchestrator.

These cover :func:`run_feature_backfill` and the ``FeatureBackfill*`` result
DTOs with an injected stub ``feature_set_backfiller`` so the orchestration
logic (date policy, lookback windows, dry-run, skip-existing) is exercised
without depending on any particular feature family.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.application.features.backfill import run_feature_backfill
from quant_platform.application.features.backfill_types import (
    FeatureBackfillDay,
    FeatureBackfillResult,
)
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector

ARTIFACT_URI = "file:///tmp/objects"


class _Repo:
    def __init__(self) -> None:
        self.vectors: list[FeatureVector] = []

    async def store_vector(self, vector: FeatureVector) -> None:
        self.vectors.append(vector)

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        as_of: datetime,
    ) -> list[FeatureVector]:
        wanted = set(instrument_ids)
        return [
            vector
            for vector in self.vectors
            if vector.instrument_id in wanted
            and vector.feature_set_version == feature_set_version
            and vector.as_of <= as_of
        ]


class _BarStore:
    def __init__(self, bars_by_instrument: dict[uuid.UUID, list[MarketBar]]) -> None:
        self._bars_by_instrument = bars_by_instrument
        self.queries: list[tuple[uuid.UUID, int, datetime, datetime]] = []

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        self.queries.append((instrument_id, bar_seconds, start, end))
        return [
            bar
            for bar in self._bars_by_instrument.get(instrument_id, [])
            if start <= bar.timestamp <= end
        ]


def _bars(instrument_id: uuid.UUID, count: int, *, end: datetime) -> list[MarketBar]:
    rows: list[MarketBar] = []
    for offset in range(count):
        timestamp = end - timedelta(days=count - 1 - offset)
        price = Decimal("100") + Decimal(offset)
        rows.append(
            MarketBar(
                bar_id=uuid.uuid4(),
                instrument_id=instrument_id,
                timestamp=timestamp,
                bar_seconds=86400,
                open=price - Decimal("0.50"),
                high=price + Decimal("1.00"),
                low=price - Decimal("1.00"),
                close=price,
                volume=1_000_000 + offset * 1_000,
            )
        )
    return rows


def _stub_backfiller(*, stored: list[FeatureVector] | None = None):
    calls: list[dict[str, object]] = []

    async def _backfiller(
        *,
        bar_history,
        feature_repo,
        strategy_run_id,
        as_of,
        feature_set_version,
        artifact_uri,
        dry_run,
        events_by_instrument=None,
    ) -> int:
        calls.append({"as_of": as_of, "instruments": tuple(bar_history), "dry_run": dry_run})
        if not dry_run:
            for instrument_id in bar_history:
                vector = FeatureVector(
                    vector_id=uuid.uuid4(),
                    instrument_id=instrument_id,
                    strategy_run_id=strategy_run_id,
                    as_of=as_of,
                    available_at=as_of,
                    feature_set_version=feature_set_version,
                    features={"stub_feature": 1.0},
                )
                await feature_repo.store_vector(vector)
                if stored is not None:
                    stored.append(vector)
        return len(bar_history)

    _backfiller.calls = calls  # type: ignore[attr-defined]
    return _backfiller


@pytest.mark.asyncio
async def test_run_feature_backfill_invokes_backfiller_and_stores_summary() -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 23, tzinfo=UTC)
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=end)})
    repo = _Repo()
    backfiller = _stub_backfiller()

    result = await run_feature_backfill(
        instruments=[SimpleNamespace(instrument_id=instrument_id)],
        bar_store=bar_store,
        feature_repo=repo,
        artifact_uri=ARTIFACT_URI,
        start=datetime(2026, 1, 22, tzinfo=UTC),
        end=end,
        feature_set_version="stub-v1",
        bar_seconds=86400,
        lookback_days=30,
        dry_run=False,
        feature_set_backfiller=backfiller,
    )

    assert isinstance(result, FeatureBackfillResult)
    assert result.date_policy == "nyse-sessions"
    assert [day.as_of for day in result.days] == [
        datetime(2026, 1, 22, tzinfo=UTC),
        datetime(2026, 1, 23, tzinfo=UTC),
    ]
    assert result.vectors_total == 2
    assert len(repo.vectors) == 2
    assert len(backfiller.calls) == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_feature_backfill_dry_run_does_not_store() -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 22, tzinfo=UTC)
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=end)})
    repo = _Repo()

    result = await run_feature_backfill(
        instruments=[SimpleNamespace(instrument_id=instrument_id)],
        bar_store=bar_store,
        feature_repo=repo,
        artifact_uri=ARTIFACT_URI,
        start=end,
        end=end,
        feature_set_version="stub-v1",
        bar_seconds=86400,
        lookback_days=30,
        dry_run=True,
        feature_set_backfiller=_stub_backfiller(),
    )

    assert result.dry_run is True
    assert result.vectors_total == 1
    assert repo.vectors == []
    assert result.to_payload()["vectors_would_store"] == 1


@pytest.mark.asyncio
async def test_run_feature_backfill_skips_instruments_without_enough_history() -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 22, tzinfo=UTC)
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 5, end=end)})
    repo = _Repo()

    result = await run_feature_backfill(
        instruments=[SimpleNamespace(instrument_id=instrument_id)],
        bar_store=bar_store,
        feature_repo=repo,
        artifact_uri=ARTIFACT_URI,
        start=end,
        end=end,
        feature_set_version="stub-v1",
        bar_seconds=86400,
        lookback_days=30,
        dry_run=False,
        feature_set_backfiller=_stub_backfiller(),
    )

    assert result.vectors_total == 0
    assert result.skipped_insufficient_history == 1
    assert repo.vectors == []


@pytest.mark.asyncio
async def test_run_feature_backfill_skips_existing_immutable_vectors() -> None:
    instrument_id = uuid.uuid4()
    as_of = datetime(2026, 1, 22, tzinfo=UTC)
    repo = _Repo()
    await repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument_id,
            strategy_run_id=uuid.uuid4(),
            as_of=as_of,
            available_at=as_of,
            feature_set_version="stub-v1",
            features={"stub_feature": 0.0},
        )
    )
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=as_of)})

    result = await run_feature_backfill(
        instruments=[SimpleNamespace(instrument_id=instrument_id)],
        bar_store=bar_store,
        feature_repo=repo,
        artifact_uri=ARTIFACT_URI,
        start=as_of,
        end=as_of,
        feature_set_version="stub-v1",
        bar_seconds=86400,
        lookback_days=30,
        dry_run=False,
        feature_set_backfiller=_stub_backfiller(),
    )

    assert result.vectors_total == 0
    assert result.skipped_existing_vectors == 1
    assert len(repo.vectors) == 1


@pytest.mark.asyncio
async def test_run_feature_backfill_without_backfiller_yields_no_vectors() -> None:
    instrument_id = uuid.uuid4()
    end = datetime(2026, 1, 22, tzinfo=UTC)
    bar_store = _BarStore({instrument_id: _bars(instrument_id, 60, end=end)})
    repo = _Repo()

    result = await run_feature_backfill(
        instruments=[SimpleNamespace(instrument_id=instrument_id)],
        bar_store=bar_store,
        feature_repo=repo,
        artifact_uri=ARTIFACT_URI,
        start=end,
        end=end,
        feature_set_version="stub-v1",
        bar_seconds=86400,
        lookback_days=30,
        dry_run=False,
        feature_set_backfiller=None,
    )

    assert result.vectors_total == 0
    assert repo.vectors == []


def test_feature_backfill_day_and_result_payloads_round_trip() -> None:
    day = FeatureBackfillDay(
        as_of=datetime(2026, 1, 22, tzinfo=UTC),
        instruments_requested=4,
        instruments_with_history=3,
        feature_vectors=3,
        skipped_insufficient_history=1,
        skipped_existing_vectors=0,
    )
    result = FeatureBackfillResult(
        feature_set_version="stub-v1",
        date_policy="nyse-sessions",
        dry_run=False,
        start=datetime(2026, 1, 22, tzinfo=UTC),
        end=datetime(2026, 1, 22, tzinfo=UTC),
        days=(day,),
    )

    payload = result.to_payload()

    assert payload["vectors_stored"] == 3
    assert payload["days_processed"] == 1
    assert payload["skipped_insufficient_history"] == 1
    assert payload["daily"][0] == day.to_payload()
