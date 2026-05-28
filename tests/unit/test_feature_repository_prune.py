"""Tests for the ``FeatureRepository.prune`` contract (Phase 4.4)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import (
    InMemoryFeatureRepository,
)


def _vec(as_of: datetime) -> FeatureVector:
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        as_of=as_of,
        feature_set_version="v1",
        features={"momentum_1m": 0.1},
        strategy_run_id=uuid.uuid4(),
    )


@pytest.mark.asyncio
async def test_prune_removes_stale_rows_only() -> None:
    repo = InMemoryFeatureRepository()
    now = datetime(2026, 4, 23, tzinfo=UTC)

    keep = _vec(now)
    stale = _vec(now - timedelta(days=10))
    await repo.store_vector(keep)
    await repo.store_vector(stale)

    deleted = await repo.prune(now - timedelta(days=5))
    assert deleted == 1

    remaining = await repo.get_vectors([keep.instrument_id, stale.instrument_id], "v1", now)
    assert {v.instrument_id for v in remaining} == {keep.instrument_id}


@pytest.mark.asyncio
async def test_prune_is_noop_when_everything_is_recent() -> None:
    repo = InMemoryFeatureRepository()
    now = datetime(2026, 4, 23, tzinfo=UTC)
    await repo.store_vector(_vec(now))

    deleted = await repo.prune(now - timedelta(days=5))
    assert deleted == 0
