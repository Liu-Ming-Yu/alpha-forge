"""In-memory V2 dataset-catalog repository."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.market_data import BarDataset, DatasetQuorumEvidence
    from quant_platform.core.domain.research import FeatureDataset


class InMemoryDatasetCatalog:
    """Dataset manifest registry for Bronze/Silver/Gold bars and features."""

    def __init__(self) -> None:
        self._bar_datasets: dict[uuid.UUID, BarDataset] = {}
        self._feature_datasets: dict[uuid.UUID, FeatureDataset] = {}
        self._quorum: dict[str, list[DatasetQuorumEvidence]] = defaultdict(list)

    async def register_bar_dataset(self, dataset: BarDataset) -> None:
        self._bar_datasets[dataset.dataset_id] = dataset

    async def register_feature_dataset(self, dataset: FeatureDataset) -> None:
        self._feature_datasets[dataset.dataset_id] = dataset

    async def latest_feature_dataset(
        self,
        feature_set_version: str,
        *,
        as_of: datetime,
        min_quality: str = "approved",
    ) -> FeatureDataset | None:
        candidates = [
            row
            for row in self._feature_datasets.values()
            if row.feature_set_version == feature_set_version
            and row.as_of <= as_of
            and row.available_at <= as_of
            and _quality_allowed(row.quality_status, min_quality)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.available_at, row.as_of))

    async def record_quorum_evidence(self, evidence: DatasetQuorumEvidence) -> None:
        rows = self._quorum[evidence.dataset_kind]
        rows[:] = [existing for existing in rows if existing.evidence_id != evidence.evidence_id]
        rows.append(evidence)
        rows.sort(key=lambda row: row.as_of)

    async def latest_quorum_evidence(
        self,
        dataset_kind: str,
        *,
        as_of: datetime,
    ) -> DatasetQuorumEvidence | None:
        candidates = [row for row in self._quorum.get(dataset_kind, []) if row.as_of <= as_of]
        return candidates[-1] if candidates else None


def _quality_allowed(actual: str, minimum: str) -> bool:
    if minimum == "approved":
        return actual == "approved"
    if minimum == "pending":
        return actual in {"pending", "approved"}
    return actual == minimum
