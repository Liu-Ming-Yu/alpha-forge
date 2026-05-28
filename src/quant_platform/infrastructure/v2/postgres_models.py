"""Postgres-backed V2 model artifact and alpha-readiness repositories."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.core.domain.research import (
    AlphaReadinessReport,
    ModelArtifact,
    ModelCard,
    PromotionState,
)
from quant_platform.infrastructure.v2.postgres_mappers import (
    _row_to_alpha_report,
    _row_to_model_artifact,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresModelArtifactRepository:
    """Postgres-backed model artifact, model-card, and alpha readiness store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register_artifact(self, artifact: ModelArtifact) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO model_artifacts_v2
                        (artifact_id, model_name, model_version, artifact_uri,
                         artifact_hash, feature_schema_hash, training_start,
                         training_end, created_at, promotion_state, rollback_artifact_id)
                    VALUES
                        (:artifact_id, :model_name, :model_version, :artifact_uri,
                         :artifact_hash, :feature_schema_hash, :training_start,
                         :training_end, :created_at, :promotion_state, :rollback_artifact_id)
                    ON CONFLICT (artifact_id) DO NOTHING
                """),
                {
                    "artifact_id": artifact.artifact_id,
                    "model_name": artifact.model_name,
                    "model_version": artifact.model_version,
                    "artifact_uri": artifact.artifact_uri,
                    "artifact_hash": artifact.artifact_hash,
                    "feature_schema_hash": artifact.feature_schema_hash,
                    "training_start": artifact.training_start,
                    "training_end": artifact.training_end,
                    "created_at": artifact.created_at,
                    "promotion_state": artifact.promotion_state.value,
                    "rollback_artifact_id": artifact.rollback_artifact_id,
                },
            )

    async def get_artifact(self, artifact_id: uuid.UUID) -> ModelArtifact | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("SELECT * FROM model_artifacts_v2 WHERE artifact_id = :artifact_id"),
                        {"artifact_id": artifact_id},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_model_artifact(row) if row else None

    async def save_model_card(self, card: ModelCard) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO model_cards
                        (card_id, artifact_id, model_name, model_version, owner,
                         intended_use, metrics_json, risk_notes, created_at)
                    VALUES
                        (:card_id, :artifact_id, :model_name, :model_version, :owner,
                         :intended_use, CAST(:metrics_json AS JSONB), :risk_notes, :created_at)
                    ON CONFLICT (card_id) DO NOTHING
                """),
                {
                    "card_id": card.card_id,
                    "artifact_id": card.artifact_id,
                    "model_name": card.model_name,
                    "model_version": card.model_version,
                    "owner": card.owner,
                    "intended_use": card.intended_use,
                    "metrics_json": json.dumps(card.metrics, default=str),
                    "risk_notes": card.risk_notes,
                    "created_at": card.created_at,
                },
            )

    async def save_alpha_readiness(self, report: AlphaReadinessReport) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO alpha_readiness_reports
                        (report_id, alpha_source, as_of, promotion_state, passed,
                         metrics_json, drift_json, rollback_target)
                    VALUES
                        (:report_id, :alpha_source, :as_of, :promotion_state, :passed,
                         CAST(:metrics_json AS JSONB), CAST(:drift_json AS JSONB),
                         :rollback_target)
                    ON CONFLICT (report_id) DO NOTHING
                """),
                {
                    "report_id": report.report_id,
                    "alpha_source": report.alpha_source,
                    "as_of": report.as_of,
                    "promotion_state": report.promotion_state.value,
                    "passed": report.passed,
                    "metrics_json": json.dumps(report.metrics, default=str),
                    "drift_json": json.dumps(report.drift, default=str),
                    "rollback_target": report.rollback_target,
                },
            )

    async def evaluate_alpha(
        self,
        source_name: str,
        *,
        as_of: datetime,
    ) -> AlphaReadinessReport:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM alpha_readiness_reports
                            WHERE alpha_source = :source_name AND as_of <= :as_of
                            ORDER BY as_of DESC
                            LIMIT 1
                        """),
                        {"source_name": source_name, "as_of": as_of},
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return AlphaReadinessReport(
                report_id=uuid.uuid4(),
                alpha_source=source_name,
                as_of=as_of,
                promotion_state=PromotionState.SHADOW,
                passed=False,
                metrics={},
                drift={},
            )
        return _row_to_alpha_report(row)
