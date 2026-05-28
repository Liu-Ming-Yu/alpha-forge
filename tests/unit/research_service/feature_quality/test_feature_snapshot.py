from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.core.domain.research import FeatureDataset, FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.infrastructure.v2.state import InMemoryDatasetCatalog
from quant_platform.services.research_service.feature_quality.snapshot import load_feature_snapshot


@pytest.mark.asyncio
async def test_load_feature_snapshot_requires_approved_dataset_and_vectors() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    instrument_id = uuid.uuid4()
    catalog = InMemoryDatasetCatalog()
    repo = InMemoryFeatureRepository()
    dataset = FeatureDataset(
        dataset_id=uuid.uuid4(),
        feature_set_version="v2",
        as_of=as_of,
        available_at=as_of,
        schema_hash="schema",
        source_dataset_ids=(uuid.uuid4(),),
        artifact_uri="s3://features/v2",
        quality_status="approved",
    )
    await catalog.register_feature_dataset(dataset)
    await repo.store_vector(
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=instrument_id,
            as_of=as_of,
            feature_set_version="v2",
            features={"momentum": 0.5},
            strategy_run_id=uuid.uuid4(),
        )
    )

    snapshot = await load_feature_snapshot(
        dataset_catalog=catalog,
        feature_repo=repo,
        instrument_ids=[instrument_id],
        feature_set_version="v2",
        as_of=as_of,
    )

    assert snapshot.dataset == dataset
    assert snapshot.as_feature_data() == {instrument_id: {"momentum": 0.5}}


@pytest.mark.asyncio
async def test_load_feature_snapshot_fails_when_vector_missing() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    catalog = InMemoryDatasetCatalog()
    await catalog.register_feature_dataset(
        FeatureDataset(
            dataset_id=uuid.uuid4(),
            feature_set_version="v2",
            as_of=as_of,
            available_at=as_of,
            schema_hash="schema",
            source_dataset_ids=(uuid.uuid4(),),
            artifact_uri="s3://features/v2",
            quality_status="approved",
        )
    )

    with pytest.raises(RuntimeError, match="missing FeatureVector"):
        await load_feature_snapshot(
            dataset_catalog=catalog,
            feature_repo=InMemoryFeatureRepository(),
            instrument_ids=[uuid.uuid4()],
            feature_set_version="v2",
            as_of=as_of,
        )
