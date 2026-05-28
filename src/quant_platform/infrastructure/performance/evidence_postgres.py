"""PostgreSQL persistence for prediction and metric evidence snapshots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.performance.mappers import (
    row_to_metric_rollup,
)
from quant_platform.infrastructure.performance.mappers import (
    row_to_prediction as _row_to_prediction,
)
from quant_platform.infrastructure.performance.prediction_status import (
    build_forecast_evidence,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        ForecastEvidence,
        MetricRollupSnapshot,
        PredictionResult,
    )


class PostgresEvidencePerformanceMixin:
    """Prediction evidence and durable metric-rollup persistence methods."""

    _engine: AsyncEngine

    async def save_metric_rollup(self, snapshot: MetricRollupSnapshot) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO metric_rollup_snapshots
                        (snapshot_id, metric_name, as_of, rollup_window, value, labels_json, source)
                    VALUES
                        (:snapshot_id, :metric_name, :as_of, :window, :value,
                         CAST(:labels AS JSONB), :source)
                    ON CONFLICT (snapshot_id)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        labels_json = EXCLUDED.labels_json,
                        source = EXCLUDED.source
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "metric_name": snapshot.metric_name,
                    "as_of": snapshot.as_of,
                    "window": snapshot.window,
                    "value": snapshot.value,
                    "labels": json.dumps(snapshot.labels, default=str),
                    "source": snapshot.source,
                },
            )

    async def list_metric_rollups(
        self,
        metric_name: str | None = None,
        *,
        limit: int = 500,
    ) -> list[MetricRollupSnapshot]:
        params: dict[str, object] = {"metric_name": metric_name, "limit": max(1, limit)}
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT snapshot_id, metric_name, as_of, rollup_window AS window, value,
                                   labels_json, source
                            FROM metric_rollup_snapshots
                            WHERE (:metric_name IS NULL OR metric_name = :metric_name)
                            ORDER BY as_of DESC
                            LIMIT :limit
                        """),
                        params,
                    )
                )
                .mappings()
                .all()
            )
        return list(reversed([row_to_metric_rollup(row) for row in rows]))

    async def save_prediction_result(self, result: PredictionResult) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO prediction_results
                        (prediction_id, strategy_run_id, instrument_id, source,
                         model_version, as_of, horizon, expected_return, rank_score,
                         confidence, feature_schema_hash, calibration_bucket,
                         blockers_json, metadata_json)
                    VALUES
                        (:prediction_id, :strategy_run_id, :instrument_id, :source,
                         :model_version, :as_of, :horizon, :expected_return, :rank_score,
                         :confidence, :feature_schema_hash, :calibration_bucket,
                         CAST(:blockers AS JSONB), CAST(:metadata AS JSONB))
                    ON CONFLICT (prediction_id)
                    DO UPDATE SET
                        expected_return = EXCLUDED.expected_return,
                        rank_score = EXCLUDED.rank_score,
                        confidence = EXCLUDED.confidence,
                        feature_schema_hash = EXCLUDED.feature_schema_hash,
                        calibration_bucket = EXCLUDED.calibration_bucket,
                        blockers_json = EXCLUDED.blockers_json,
                        metadata_json = EXCLUDED.metadata_json
                """),
                {
                    "prediction_id": result.prediction_id,
                    "strategy_run_id": result.strategy_run_id,
                    "instrument_id": result.instrument_id,
                    "source": result.source,
                    "model_version": result.model_version,
                    "as_of": result.as_of,
                    "horizon": result.horizon,
                    "expected_return": result.expected_return,
                    "rank_score": result.rank_score,
                    "confidence": result.confidence,
                    "feature_schema_hash": result.feature_schema_hash,
                    "calibration_bucket": result.calibration_bucket,
                    "blockers": json.dumps(list(result.blockers)),
                    "metadata": json.dumps(result.metadata, default=str),
                },
            )

    async def list_prediction_results(
        self,
        *,
        source: str | None = None,
        model_version: str | None = None,
        strategy_run_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        as_of: datetime | None = None,
        limit: int = 500,
    ) -> list[PredictionResult]:
        params: dict[str, object] = {
            "source": source,
            "model_version": model_version,
            "strategy_run_id": strategy_run_id,
            "instrument_id": instrument_id,
            "as_of": as_of,
            "limit": max(1, limit),
        }
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                            SELECT prediction_id, strategy_run_id, instrument_id, source,
                                   model_version, as_of, horizon, expected_return, rank_score,
                                   confidence, feature_schema_hash, calibration_bucket,
                                   blockers_json, metadata_json
                            FROM prediction_results
                            WHERE (
                                  CAST(:source AS TEXT) IS NULL
                                  OR source = CAST(:source AS TEXT)
                              )
                              AND (
                                  CAST(:model_version AS TEXT) IS NULL
                                  OR model_version = CAST(:model_version AS TEXT)
                              )
                              AND (
                                  CAST(:strategy_run_id AS UUID) IS NULL
                                  OR strategy_run_id = CAST(:strategy_run_id AS UUID)
                              )
                              AND (
                                  CAST(:instrument_id AS UUID) IS NULL
                                  OR instrument_id = CAST(:instrument_id AS UUID)
                              )
                              AND (
                                  CAST(:as_of AS TIMESTAMPTZ) IS NULL
                                  OR as_of <= CAST(:as_of AS TIMESTAMPTZ)
                              )
                            ORDER BY as_of DESC
                            LIMIT :limit
                        """),
                        params,
                    )
                )
                .mappings()
                .all()
            )
        return list(reversed([_row_to_prediction(row) for row in rows]))

    async def forecast_evidence(
        self,
        source: str,
        *,
        model_version: str | None = None,
        as_of: datetime,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> ForecastEvidence:
        rows = await self.list_prediction_results(
            source=source,
            model_version=model_version,
            as_of=as_of,
            limit=limit,
        )
        return build_forecast_evidence(
            source,
            model_version=model_version,
            as_of=as_of,
            rows=rows,
            stale_after_hours=stale_after_hours,
            min_confidence=min_confidence,
        )
