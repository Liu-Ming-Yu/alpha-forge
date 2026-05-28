"""Persistence for governed feature audit results."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.core.domain.research import FeatureAuditResult, FeatureProductionState
from quant_platform.infrastructure.postgres.row_coercion import (
    require_float,
    require_mapping,
    require_sequence,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class InMemoryFeatureAuditRepository:
    """In-memory feature audit state for tests and local shadow runs."""

    def __init__(self) -> None:
        self._rows: dict[uuid.UUID, FeatureAuditResult] = {}

    async def save_feature_audit(self, result: FeatureAuditResult) -> None:
        self._rows[result.audit_id] = result

    async def latest_feature_audit(
        self,
        feature_name: str,
        feature_version: str | None = None,
    ) -> FeatureAuditResult | None:
        rows = [
            row
            for row in self._rows.values()
            if row.feature_name == feature_name
            and (feature_version is None or row.feature_version == feature_version)
        ]
        return max(rows, key=lambda row: row.as_of) if rows else None

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[FeatureAuditResult]:
        rows = list(self._rows.values())
        if feature_name is not None:
            rows = [row for row in rows if row.feature_name == feature_name]
        rows.sort(key=lambda row: row.as_of, reverse=True)
        return rows[: max(1, limit)]


class PostgresFeatureAuditRepository:
    """PostgreSQL-backed feature audit state."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_feature_audit(self, result: FeatureAuditResult) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO feature_audit_results
                        (audit_id, feature_name, feature_version, feature_set_version,
                         as_of, sample_start, sample_end, status, passed,
                         metrics_json, gate_results_json, artifact_uri,
                         schema_hash, code_commit, blockers_json)
                    VALUES
                        (:audit_id, :feature_name, :feature_version, :feature_set_version,
                         :as_of, :sample_start, :sample_end, :status, :passed,
                         CAST(:metrics_json AS JSONB), CAST(:gate_results_json AS JSONB),
                         :artifact_uri, :schema_hash, :code_commit,
                         CAST(:blockers_json AS JSONB))
                    ON CONFLICT (audit_id) DO NOTHING
                """),
                _params(result),
            )

    async def latest_feature_audit(
        self,
        feature_name: str,
        feature_version: str | None = None,
    ) -> FeatureAuditResult | None:
        params: dict[str, object] = {"feature_name": feature_name}
        if feature_version is not None:
            params["feature_version"] = feature_version
            query = text("""
                        SELECT audit_id, feature_name, feature_version,
                               feature_set_version, as_of, sample_start, sample_end,
                               status, passed, metrics_json, gate_results_json,
                               artifact_uri, schema_hash, code_commit, blockers_json
                        FROM feature_audit_results
                        WHERE feature_name = :feature_name
                          AND feature_version = :feature_version
                        ORDER BY as_of DESC
                        LIMIT 1
                        """)
        else:
            query = text("""
                        SELECT audit_id, feature_name, feature_version,
                               feature_set_version, as_of, sample_start, sample_end,
                               status, passed, metrics_json, gate_results_json,
                               artifact_uri, schema_hash, code_commit, blockers_json
                        FROM feature_audit_results
                        WHERE feature_name = :feature_name
                        ORDER BY as_of DESC
                        LIMIT 1
                        """)
        async with self._engine.connect() as conn:
            row = (await conn.execute(query, params)).mappings().first()
        return _row_to_result(row) if row is not None else None

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[FeatureAuditResult]:
        params: dict[str, object] = {"limit": max(1, limit)}
        if feature_name is not None:
            params["feature_name"] = feature_name
            query = text("""
                        SELECT audit_id, feature_name, feature_version,
                               feature_set_version, as_of, sample_start, sample_end,
                               status, passed, metrics_json, gate_results_json,
                               artifact_uri, schema_hash, code_commit, blockers_json
                        FROM feature_audit_results
                        WHERE feature_name = :feature_name
                        ORDER BY as_of DESC
                        LIMIT :limit
                        """)
        else:
            query = text("""
                        SELECT audit_id, feature_name, feature_version,
                               feature_set_version, as_of, sample_start, sample_end,
                               status, passed, metrics_json, gate_results_json,
                               artifact_uri, schema_hash, code_commit, blockers_json
                        FROM feature_audit_results
                        ORDER BY as_of DESC
                        LIMIT :limit
                        """)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()
        return [_row_to_result(row) for row in rows]


def build_feature_audit_repository(
    dsn: str,
    engine: AsyncEngine | None = None,
) -> InMemoryFeatureAuditRepository | PostgresFeatureAuditRepository:
    if not dsn:
        return InMemoryFeatureAuditRepository()
    if engine is None:
        from quant_platform.infrastructure.postgres.repositories import create_pg_engine

        engine = create_pg_engine(dsn)
    return PostgresFeatureAuditRepository(engine)


def feature_audit_payload(result: FeatureAuditResult) -> dict[str, object]:
    """JSON-safe representation for CLI/API responses."""
    return {
        "audit_id": str(result.audit_id),
        "feature_name": result.feature_name,
        "feature_version": result.feature_version,
        "feature_set_version": result.feature_set_version,
        "as_of": result.as_of.isoformat(),
        "sample_start": result.sample_start.isoformat(),
        "sample_end": result.sample_end.isoformat(),
        "status": result.status.value,
        "passed": result.passed,
        "metrics": dict(result.metrics),
        "gate_results": dict(result.gate_results),
        "artifact_uri": result.artifact_uri,
        "schema_hash": result.schema_hash,
        "code_commit": result.code_commit,
        "blockers": list(result.blockers),
    }


def _params(result: FeatureAuditResult) -> dict[str, object]:
    return {
        "audit_id": result.audit_id,
        "feature_name": result.feature_name,
        "feature_version": result.feature_version,
        "feature_set_version": result.feature_set_version,
        "as_of": result.as_of,
        "sample_start": result.sample_start,
        "sample_end": result.sample_end,
        "status": result.status.value,
        "passed": result.passed,
        "metrics_json": json.dumps(dict(result.metrics), sort_keys=True),
        "gate_results_json": json.dumps(dict(result.gate_results), sort_keys=True),
        "artifact_uri": result.artifact_uri,
        "schema_hash": result.schema_hash,
        "code_commit": result.code_commit,
        "blockers_json": json.dumps(list(result.blockers), sort_keys=True),
    }


def _row_to_result(row: object) -> FeatureAuditResult:
    data = require_mapping(row, name="feature_audit_row")
    metrics = data["metrics_json"]
    gates = data["gate_results_json"]
    blockers = data["blockers_json"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    if isinstance(gates, str):
        gates = json.loads(gates)
    if isinstance(blockers, str):
        blockers = json.loads(blockers)
    metrics_map = require_mapping(metrics, name="feature_audit.metrics_json")
    gates_map = require_mapping(gates, name="feature_audit.gate_results_json")
    blocker_items = require_sequence(blockers, name="feature_audit.blockers_json")
    return FeatureAuditResult(
        audit_id=uuid.UUID(str(data["audit_id"])),
        feature_name=str(data["feature_name"]),
        feature_version=str(data["feature_version"]),
        feature_set_version=str(data["feature_set_version"]),
        as_of=_dt(data["as_of"]),
        sample_start=_dt(data["sample_start"]),
        sample_end=_dt(data["sample_end"]),
        status=FeatureProductionState(str(data["status"])),
        passed=bool(data["passed"]),
        metrics={
            str(k): require_float(v, name=f"metrics_json.{k}") for k, v in metrics_map.items()
        },
        gate_results={str(k): bool(v) for k, v in gates_map.items()},
        artifact_uri=str(data["artifact_uri"]),
        schema_hash=str(data["schema_hash"]),
        code_commit=str(data["code_commit"]),
        blockers=tuple(str(item) for item in blocker_items),
    )


def _dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
