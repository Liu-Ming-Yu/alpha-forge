"""FeatureSnapshot loading helpers for V2 live/research parity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.research import FeatureSnapshot

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.contracts import DatasetCatalog, FeatureRepository


async def load_feature_snapshot(
    *,
    dataset_catalog: DatasetCatalog,
    feature_repo: FeatureRepository,
    instrument_ids: list[uuid.UUID],
    feature_set_version: str,
    as_of: datetime,
) -> FeatureSnapshot:
    """Load an approved FeatureDataset and all required vectors.

    Raises RuntimeError when the approved dataset or any required instrument
    vector is missing.  This is the V2 fail-closed replacement for ad hoc
    ``feature_data`` in live paths.
    """
    dataset = await dataset_catalog.latest_feature_dataset(
        feature_set_version,
        as_of=as_of,
        min_quality="approved",
    )
    if dataset is None:
        raise RuntimeError(
            f"missing approved FeatureDataset version={feature_set_version} as_of={as_of}"
        )
    vectors = await feature_repo.get_vectors(instrument_ids, feature_set_version, as_of)
    found = {vector.instrument_id for vector in vectors}
    missing = sorted(str(instrument_id) for instrument_id in set(instrument_ids) - found)
    if missing:
        raise RuntimeError(
            "missing FeatureVector(s) for approved FeatureDataset: " + ", ".join(missing)
        )
    return FeatureSnapshot(dataset=dataset, vectors=tuple(vectors), as_of=as_of)
