"""PostgreSQL-backed FeatureRepository implementation.

Satisfies the ``FeatureRepository`` protocol with durable, immutable storage.
Feature vectors are keyed by (instrument_id, feature_set_version, as_of) and
cannot be overwritten once stored.

Artifact URIs are stored alongside each vector for feature lineage — callers
can trace exactly which bar data produced the features that drove a live trade.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import bindparam, text

from quant_platform.core.domain.research import FeatureVector

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)


class PostgresFeatureRepository:
    """Durable feature vector store backed by PostgreSQL.

    Requires the ``feature_vectors`` table created by Alembic migration 002.

    Must never:
        Overwrite an existing vector for the same natural key.
        Return vectors whose as_of or available_at is after the requested as_of.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def store_vector(self, vector: FeatureVector) -> None:
        features_json = json.dumps(
            {k: v for k, v in vector.features.items()},
            default=str,
        )
        artifact_uri = ""
        if hasattr(vector, "artifact_uri"):
            artifact_uri = getattr(vector, "artifact_uri", "")
        available_at = vector.available_at or vector.as_of

        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("""
                    INSERT INTO feature_vectors
                        (vector_id, instrument_id, as_of, feature_set_version,
                         features, strategy_run_id, artifact_uri, available_at)
                    VALUES
                        (:vector_id, :instrument_id, :as_of, :feature_set_version,
                         :features, :strategy_run_id, :artifact_uri, :available_at)
                    ON CONFLICT ON CONSTRAINT uq_feature_vector_natural_key DO NOTHING
                """),
                {
                    "vector_id": vector.vector_id,
                    "instrument_id": vector.instrument_id,
                    "as_of": vector.as_of,
                    "feature_set_version": vector.feature_set_version,
                    "features": features_json,
                    "strategy_run_id": vector.strategy_run_id,
                    "artifact_uri": artifact_uri,
                    "available_at": available_at,
                },
            )
        if int(getattr(result, "rowcount", 0) or 0) == 0:
            raise ValueError(
                f"Duplicate FeatureVector: instrument={vector.instrument_id}, "
                f"version={vector.feature_set_version}, as_of={vector.as_of}"
            )

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        as_of: datetime,
    ) -> list[FeatureVector]:
        if not instrument_ids:
            return []

        params: dict[str, object] = {"instrument_ids": tuple(instrument_ids)}
        params["version"] = feature_set_version
        params["as_of"] = as_of

        query = text("""
            SELECT DISTINCT ON (instrument_id)
                vector_id, instrument_id, as_of, feature_set_version,
                features, strategy_run_id, artifact_uri, available_at
            FROM feature_vectors
            WHERE instrument_id IN :instrument_ids
              AND feature_set_version = :version
              AND as_of <= :as_of
              AND available_at <= :as_of
            ORDER BY instrument_id, as_of DESC, available_at DESC
        """).bindparams(bindparam("instrument_ids", expanding=True))

        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()

        result: list[FeatureVector] = []
        for row in rows:
            features = row["features"]
            if isinstance(features, str):
                features = json.loads(features)
            result.append(
                FeatureVector(
                    vector_id=uuid.UUID(str(row["vector_id"])),
                    instrument_id=uuid.UUID(str(row["instrument_id"])),
                    as_of=row["as_of"],
                    feature_set_version=row["feature_set_version"],
                    features=features,
                    strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
                    artifact_uri=row.get("artifact_uri", "") or "",
                    available_at=row["available_at"],
                )
            )

        return result

    async def get_vector_history(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        start: datetime,
        end: datetime,
    ) -> list[FeatureVector]:
        """Return all visible vectors in a closed point-in-time window."""
        if not instrument_ids:
            return []

        query = text("""
            SELECT vector_id, instrument_id, as_of, feature_set_version,
                   features, strategy_run_id, artifact_uri, available_at
            FROM feature_vectors
            WHERE instrument_id IN :instrument_ids
              AND feature_set_version = :version
              AND as_of >= :start
              AND as_of <= :end
              AND available_at <= :end
            ORDER BY instrument_id, as_of, available_at
        """).bindparams(bindparam("instrument_ids", expanding=True))
        params: dict[str, object] = {
            "instrument_ids": tuple(instrument_ids),
            "version": feature_set_version,
            "start": start,
            "end": end,
        }

        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()

        result: list[FeatureVector] = []
        for row in rows:
            features = row["features"]
            if isinstance(features, str):
                features = json.loads(features)
            result.append(
                FeatureVector(
                    vector_id=uuid.UUID(str(row["vector_id"])),
                    instrument_id=uuid.UUID(str(row["instrument_id"])),
                    as_of=row["as_of"],
                    feature_set_version=row["feature_set_version"],
                    features=features,
                    strategy_run_id=uuid.UUID(str(row["strategy_run_id"])),
                    artifact_uri=row.get("artifact_uri", "") or "",
                    available_at=row["available_at"],
                )
            )
        return result

    async def prune(self, older_than: datetime) -> int:
        """Delete feature rows with ``as_of < older_than``.

        Retires R-DAT-03: provides a durable retention hook that
        ``quant-platform features retention`` exercises.  Returns the
        number of rows deleted.
        """
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM feature_vectors WHERE as_of < :cutoff"),
                {"cutoff": older_than},
            )
        deleted = int(getattr(result, "rowcount", 0) or 0)
        log.info(
            "feature_repository.pruned",
            cutoff=older_than.isoformat(),
            deleted=deleted,
        )
        return deleted
