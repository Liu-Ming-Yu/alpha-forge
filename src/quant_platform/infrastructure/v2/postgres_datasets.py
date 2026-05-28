"""Postgres-backed V2 dataset catalog repository."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.v2.postgres_mappers import (
    _row_to_feature_dataset,
    _row_to_quorum,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.market_data import BarDataset, DatasetQuorumEvidence
    from quant_platform.core.domain.research import FeatureDataset


class PostgresDatasetCatalog:
    """Postgres-backed dataset catalog and vendor quorum evidence."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register_bar_dataset(self, dataset: BarDataset) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO bar_datasets
                        (dataset_id, layer, vendor, bar_seconds, start_at, end_at,
                         as_of, available_at, schema_hash, source_uri,
                         quality_status, row_count)
                    VALUES
                        (:dataset_id, :layer, :vendor, :bar_seconds, :start_at, :end_at,
                         :as_of, :available_at, :schema_hash, :source_uri,
                         :quality_status, :row_count)
                    ON CONFLICT (dataset_id) DO NOTHING
                """),
                {
                    "dataset_id": dataset.dataset_id,
                    "layer": dataset.layer.value,
                    "vendor": dataset.vendor,
                    "bar_seconds": dataset.bar_seconds,
                    "start_at": dataset.start_at,
                    "end_at": dataset.end_at,
                    "as_of": dataset.as_of,
                    "available_at": dataset.available_at,
                    "schema_hash": dataset.schema_hash,
                    "source_uri": dataset.source_uri,
                    "quality_status": dataset.quality.value,
                    "row_count": dataset.row_count,
                },
            )

    async def register_feature_dataset(self, dataset: FeatureDataset) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO feature_datasets
                        (dataset_id, feature_set_version, as_of, available_at,
                         schema_hash, source_dataset_ids_json, artifact_uri,
                         quality_status)
                    VALUES
                        (:dataset_id, :feature_set_version, :as_of, :available_at,
                         :schema_hash, CAST(:source_dataset_ids_json AS JSONB),
                         :artifact_uri, :quality_status)
                    ON CONFLICT (dataset_id) DO NOTHING
                """),
                {
                    "dataset_id": dataset.dataset_id,
                    "feature_set_version": dataset.feature_set_version,
                    "as_of": dataset.as_of,
                    "available_at": dataset.available_at,
                    "schema_hash": dataset.schema_hash,
                    "source_dataset_ids_json": json.dumps(
                        [str(item) for item in dataset.source_dataset_ids]
                    ),
                    "artifact_uri": dataset.artifact_uri,
                    "quality_status": dataset.quality_status,
                },
            )

    async def latest_feature_dataset(
        self,
        feature_set_version: str,
        *,
        as_of: datetime,
        min_quality: str = "approved",
    ) -> FeatureDataset | None:
        include_pending = min_quality != "approved"
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM feature_datasets
                            WHERE feature_set_version = :feature_set_version
                              AND as_of <= :as_of
                              AND available_at <= :as_of
                              AND (
                                quality_status = 'approved'
                                OR (:include_pending AND quality_status = 'pending')
                              )
                            ORDER BY available_at DESC, as_of DESC
                            LIMIT 1
                        """),
                        {
                            "feature_set_version": feature_set_version,
                            "as_of": as_of,
                            "include_pending": include_pending,
                        },
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_feature_dataset(row) if row else None

    async def record_quorum_evidence(self, evidence: DatasetQuorumEvidence) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO dataset_quorum_evidence
                        (evidence_id, dataset_kind, as_of, vendors_json, passed,
                         required_vendor_count, max_disagreement_bps, details_json)
                    VALUES
                        (:evidence_id, :dataset_kind, :as_of, CAST(:vendors_json AS JSONB),
                         :passed, :required_vendor_count, :max_disagreement_bps,
                         CAST(:details_json AS JSONB))
                    ON CONFLICT (evidence_id) DO NOTHING
                """),
                {
                    "evidence_id": evidence.evidence_id,
                    "dataset_kind": evidence.dataset_kind,
                    "as_of": evidence.as_of,
                    "vendors_json": json.dumps(list(evidence.vendors)),
                    "passed": evidence.passed,
                    "required_vendor_count": evidence.required_vendor_count,
                    "max_disagreement_bps": evidence.max_disagreement_bps,
                    "details_json": json.dumps(evidence.details, default=str),
                },
            )

    async def latest_quorum_evidence(
        self,
        dataset_kind: str,
        *,
        as_of: datetime,
    ) -> DatasetQuorumEvidence | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM dataset_quorum_evidence
                            WHERE dataset_kind = :dataset_kind AND as_of <= :as_of
                            ORDER BY as_of DESC
                            LIMIT 1
                        """),
                        {"dataset_kind": dataset_kind, "as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_quorum(row) if row else None
