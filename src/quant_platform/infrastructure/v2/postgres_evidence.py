"""Postgres-backed V2 production/operator evidence repositories."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.v2.postgres_mappers import (
    _row_to_operator_action,
    _row_to_operator_api_key,
    _row_to_readiness_snapshot,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.domain.production import (
        AlertEvent,
        OperatorAction,
        OperatorApiKey,
        ReadinessSnapshot,
        RunbookEvidence,
    )


class PostgresProductionEvidenceRepository:
    """Postgres-backed operator, runbook, alert, readiness, and RBAC evidence."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_operator_action(self, action: OperatorAction) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO operator_actions
                        (id, occurred_at, action_type, actor, reason, metadata)
                    VALUES
                        (:id, :occurred_at, :action_type, :actor, :reason,
                         CAST(:metadata AS JSONB))
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": action.action_id,
                    "occurred_at": action.occurred_at,
                    "action_type": action.action_type,
                    "actor": action.actor,
                    "reason": action.reason,
                    "metadata": json.dumps(action.metadata, default=str),
                },
            )

    async def list_operator_actions(
        self,
        *,
        action_type: str | None = None,
        limit: int = 200,
    ) -> list[OperatorAction]:
        params: dict[str, object] = {"limit": max(1, limit)}
        if action_type:
            params["action_type"] = action_type
            query = text("""
                            SELECT id, occurred_at, action_type, actor, reason, metadata
                            FROM operator_actions
                            WHERE action_type = :action_type
                            ORDER BY occurred_at DESC
                            LIMIT :limit
                        """)
        else:
            query = text("""
                            SELECT id, occurred_at, action_type, actor, reason, metadata
                            FROM operator_actions
                            ORDER BY occurred_at DESC
                            LIMIT :limit
                        """)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()
        return [_row_to_operator_action(row) for row in rows]

    async def save_runbook_evidence(self, evidence: RunbookEvidence) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO runbook_evidence
                        (evidence_id, runbook_name, executed_at, actor, status,
                         artifact_uri, metadata_json)
                    VALUES
                        (:evidence_id, :runbook_name, :executed_at, :actor, :status,
                         :artifact_uri, CAST(:metadata_json AS JSONB))
                    ON CONFLICT (evidence_id) DO NOTHING
                """),
                {
                    "evidence_id": evidence.evidence_id,
                    "runbook_name": evidence.runbook_name,
                    "executed_at": evidence.executed_at,
                    "actor": evidence.actor,
                    "status": evidence.status,
                    "artifact_uri": evidence.artifact_uri,
                    "metadata_json": json.dumps(evidence.metadata, default=str),
                },
            )

    async def save_alert_event(self, event: AlertEvent) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO alert_events
                        (alert_id, occurred_at, severity, component, message,
                         resolved_at, metadata_json)
                    VALUES
                        (:alert_id, :occurred_at, :severity, :component, :message,
                         :resolved_at, CAST(:metadata_json AS JSONB))
                    ON CONFLICT (alert_id) DO NOTHING
                """),
                {
                    "alert_id": event.alert_id,
                    "occurred_at": event.occurred_at,
                    "severity": event.severity,
                    "component": event.component,
                    "message": event.message,
                    "resolved_at": event.resolved_at,
                    "metadata_json": json.dumps(event.metadata, default=str),
                },
            )

    async def save_readiness_snapshot(self, snapshot: ReadinessSnapshot) -> None:
        checks = [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "severity": check.severity,
            }
            for check in snapshot.checks
        ]
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO readiness_snapshots
                        (snapshot_id, profile, generated_at, state, passed, checks_json)
                    VALUES
                        (:snapshot_id, :profile, :generated_at, :state, :passed,
                         CAST(:checks_json AS JSONB))
                    ON CONFLICT (snapshot_id) DO NOTHING
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "profile": snapshot.profile.value,
                    "generated_at": snapshot.generated_at,
                    "state": snapshot.state.value,
                    "passed": snapshot.passed,
                    "checks_json": json.dumps(checks, default=str),
                },
            )

    async def latest_readiness_snapshot(self, profile: str) -> ReadinessSnapshot | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""
                            SELECT *
                            FROM readiness_snapshots
                            WHERE profile = :profile
                            ORDER BY generated_at DESC
                            LIMIT 1
                        """),
                        {"profile": profile},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_readiness_snapshot(row) if row else None

    async def save_api_key(self, key: OperatorApiKey) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO operator_api_keys
                        (key_id, key_hash, role, label, created_at, created_by, revoked_at)
                    VALUES
                        (:key_id, :key_hash, :role, :label, :created_at, :created_by, :revoked_at)
                    ON CONFLICT (key_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        label = EXCLUDED.label,
                        revoked_at = EXCLUDED.revoked_at
                """),
                {
                    "key_id": key.key_id,
                    "key_hash": key.key_hash,
                    "role": key.role,
                    "label": key.label,
                    "created_at": key.created_at,
                    "created_by": key.created_by,
                    "revoked_at": key.revoked_at,
                },
            )

    async def get_api_key_by_hash(self, key_hash: str) -> OperatorApiKey | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("SELECT * FROM operator_api_keys WHERE key_hash = :key_hash"),
                        {"key_hash": key_hash},
                    )
                )
                .mappings()
                .first()
            )
        return _row_to_operator_api_key(row) if row else None

    async def revoke_api_key(
        self,
        key_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    UPDATE operator_api_keys
                    SET revoked_at = :revoked_at
                    WHERE key_id = :key_id
                """),
                {"key_id": key_id, "revoked_at": revoked_at},
            )
