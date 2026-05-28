from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository

_UTC = UTC


@pytest.mark.asyncio
async def test_in_memory_feature_repository_excludes_late_available_at() -> None:
    repo = InMemoryFeatureRepository()
    instrument_id = uuid.uuid4()
    version = "available-at-test"
    decision_time = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)
    late = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=decision_time,
        feature_set_version=version,
        features={"momentum_1m": 1.0},
        strategy_run_id=uuid.uuid4(),
        available_at=decision_time + timedelta(minutes=1),
    )
    visible = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=decision_time - timedelta(minutes=1),
        feature_set_version=version,
        features={"momentum_1m": 0.5},
        strategy_run_id=uuid.uuid4(),
        available_at=decision_time,
    )

    await repo.store_vector(late)
    await repo.store_vector(visible)

    result = await repo.get_vectors([instrument_id], version, decision_time)

    assert [vector.vector_id for vector in result] == [visible.vector_id]


@pytest.mark.asyncio
async def test_feature_vector_may_be_available_before_decision_time() -> None:
    repo = InMemoryFeatureRepository()
    instrument_id = uuid.uuid4()
    version = "intraday-pit"
    decision_time = datetime(2026, 1, 3, 0, 0, tzinfo=_UTC)
    available_at = decision_time - timedelta(hours=3)
    vector = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=decision_time,
        feature_set_version=version,
        features={"opening_drive_confirmation_1d_decay": 0.4},
        strategy_run_id=uuid.uuid4(),
        available_at=available_at,
    )

    await repo.store_vector(vector)
    result = await repo.get_vectors([instrument_id], version, decision_time)

    assert result == [vector]
    assert result[0].available_at == available_at
