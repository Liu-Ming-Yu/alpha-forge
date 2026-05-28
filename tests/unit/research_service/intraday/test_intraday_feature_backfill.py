from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.research.intraday.feature_backfill_ops import (
    _sample_free_intraday_samples,
)
from quant_platform.services.research_service.intraday.candidates.screening import (
    intraday_candidates_for_set,
)
from quant_platform.services.research_service.intraday.features.backfill import (
    backfill_intraday_feature_vectors,
    feature_names_from_family_file,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path

_FEATURES = (
    "opening_drive_confirmation_1d_decay",
    "close_pressure_continuation_1d_decay",
    "vwap_accumulation_pressure_1d_decay",
)


@pytest.mark.asyncio
async def test_intraday_backfill_sets_available_at_before_or_equal_as_of() -> None:
    repo = InMemoryFeatureRepository()
    instrument_id = uuid.uuid4()
    decision = datetime(2026, 1, 3, tzinfo=UTC)

    result = await backfill_intraday_feature_vectors(
        samples=(_sample(instrument_id, decision),),
        intraday_bars=_session(instrument_id, decision - timedelta(days=1)),
        candidates=intraday_candidates_for_set("seed"),
        feature_names=_FEATURES,
        repo=repo,
        strategy_run_id=uuid.uuid4(),
        feature_set_version="paper-alpha-intraday-microstructure-v1",
        artifact_uri="file:///tmp/intraday-family.json",
    )
    vectors = await repo.get_vectors(
        [instrument_id],
        "paper-alpha-intraday-microstructure-v1",
        decision,
    )

    assert result.vector_count == 1
    assert len(vectors) == 1
    assert vectors[0].as_of == decision
    assert vectors[0].available_at is not None
    assert vectors[0].available_at <= decision
    assert set(vectors[0].features) == set(_FEATURES)


@pytest.mark.asyncio
async def test_intraday_backfill_excludes_future_only_bars() -> None:
    repo = InMemoryFeatureRepository()
    instrument_id = uuid.uuid4()
    decision = datetime(2026, 1, 3, tzinfo=UTC)

    result = await backfill_intraday_feature_vectors(
        samples=(_sample(instrument_id, decision),),
        intraday_bars=_session(instrument_id, decision + timedelta(days=1)),
        candidates=intraday_candidates_for_set("seed"),
        feature_names=_FEATURES,
        repo=repo,
        strategy_run_id=uuid.uuid4(),
        feature_set_version="paper-alpha-intraday-microstructure-v1",
        artifact_uri="file:///tmp/intraday-family.json",
    )

    assert result.vector_count == 0
    assert result.skipped_no_intraday_session == 1
    assert (
        await repo.get_vectors([instrument_id], "paper-alpha-intraday-microstructure-v1", decision)
        == []
    )


@pytest.mark.asyncio
async def test_intraday_backfill_skips_existing_immutable_vectors() -> None:
    repo = InMemoryFeatureRepository()
    instrument_id = uuid.uuid4()
    decision = datetime(2026, 1, 3, tzinfo=UTC)
    feature_set_version = "paper-alpha-intraday-microstructure-v1"
    await repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument_id,
            as_of=decision,
            feature_set_version=feature_set_version,
            features={_FEATURES[0]: 0.2},
            strategy_run_id=uuid.uuid4(),
            artifact_uri="file:///tmp/intraday-family.json",
            available_at=decision,
        )
    )

    result = await backfill_intraday_feature_vectors(
        samples=(_sample(instrument_id, decision),),
        intraday_bars=_session(instrument_id, decision - timedelta(days=1)),
        candidates=intraday_candidates_for_set("seed"),
        feature_names=_FEATURES,
        repo=repo,
        strategy_run_id=uuid.uuid4(),
        feature_set_version=feature_set_version,
        artifact_uri="file:///tmp/intraday-family.json",
    )

    vectors = await repo.get_vectors([instrument_id], feature_set_version, decision)
    assert result.vector_count == 0
    assert result.skipped_existing_vectors == 1
    assert len(vectors) == 1


def test_feature_names_from_family_file_reads_unique_members(tmp_path: Path) -> None:
    path = tmp_path / "family.json"
    path.write_text(
        json.dumps(
            {
                "feature_set_version": "paper-alpha-intraday-microstructure-v1",
                "families": {"a": [_FEATURES[0]], "b": [_FEATURES[0], _FEATURES[1]]},
            }
        ),
        encoding="utf-8",
    )

    assert feature_names_from_family_file(path) == tuple(sorted((_FEATURES[0], _FEATURES[1])))


@pytest.mark.asyncio
async def test_sample_free_intraday_backfill_builds_current_decision_samples() -> None:
    repo = InMemoryFeatureRepository()
    first = uuid.uuid5(uuid.NAMESPACE_URL, "aapl")
    second = uuid.uuid5(uuid.NAMESPACE_URL, "msft")
    decision = datetime(2026, 1, 5, tzinfo=UTC)
    await repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=first,
            as_of=decision,
            feature_set_version="paper-alpha-catalyst-v10",
            features={"text_alpha": 0.25},
            strategy_run_id=uuid.uuid4(),
            artifact_uri="file:///tmp/text.parquet",
            available_at=decision,
        )
    )

    samples, payload = await _sample_free_intraday_samples(
        feature_repo=repo,
        contracts={first: {"symbol": "AAPL"}, second: {"symbol": "MSFT"}},
        start=decision,
        end=decision,
        date_policy="nyse-sessions",
        context_feature_set_version="paper-alpha-catalyst-v10",
    )

    assert len(samples) == 2
    assert {sample.instrument_id for sample in samples} == {first, second}
    assert all(sample.as_of == decision for sample in samples)
    assert all(sample.forward_return == 0.0 for sample in samples)
    assert all(
        sample.metadata_dict()["label_mode"] == "sample_free_current_evidence" for sample in samples
    )
    assert [sample.features for sample in samples if sample.instrument_id == first] == [
        {"text_alpha": 0.25}
    ]
    assert payload["mode"] == "sample_free"
    assert payload["as_of_dates"] == 1
    assert payload["samples"] == 2
    assert payload["context_vectors_used"] == 1
    assert payload["context_vectors_missing"] == 1


def _sample(instrument_id: uuid.UUID, as_of: datetime) -> SupervisedAlphaSample:
    return SupervisedAlphaSample(
        as_of=as_of,
        instrument_id=instrument_id,
        features={},
        forward_return=0.01,
    )


def _session(instrument_id: uuid.UUID, session_day: datetime) -> tuple[MarketBar, ...]:
    start = session_day.replace(hour=14, minute=30, second=0, microsecond=0)
    return (
        _bar(instrument_id, start, "100.00", "100.40"),
        _bar(instrument_id, start + timedelta(minutes=30), "100.40", "100.70"),
        _bar(instrument_id, start + timedelta(minutes=389), "100.70", "101.20"),
    )


def _bar(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    open_price: str,
    close_price: str,
) -> MarketBar:
    open_decimal = Decimal(open_price)
    close_decimal = Decimal(close_price)
    high = max(open_decimal, close_decimal) + Decimal("0.10")
    low = min(open_decimal, close_decimal) - Decimal("0.10")
    return MarketBar(
        bar_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{instrument_id}:{timestamp.isoformat()}"),
        instrument_id=instrument_id,
        timestamp=timestamp,
        bar_seconds=60,
        open=open_decimal,
        high=high,
        low=low,
        close=close_decimal,
        volume=1000,
        vwap=(open_decimal + close_decimal) / Decimal("2"),
    )
