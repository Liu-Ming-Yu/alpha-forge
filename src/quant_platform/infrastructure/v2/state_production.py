"""In-memory V2 production-evidence repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    AlertEvent,
    OperatorAction,
    OperatorApiKey,
    ReadinessSnapshot,
    RunbookEvidence,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class InMemoryProductionEvidenceRepository:
    """Runbook, alert, readiness, and operator evidence repository."""

    def __init__(self) -> None:
        self._actions: list[OperatorAction] = []
        self._runbook: list[RunbookEvidence] = []
        self._alerts: list[AlertEvent] = []
        self._readiness: list[ReadinessSnapshot] = []
        self._api_keys: dict[uuid.UUID, OperatorApiKey] = {}

    async def record_operator_action(self, action: OperatorAction) -> None:
        self._actions = [
            existing for existing in self._actions if existing.action_id != action.action_id
        ]
        self._actions.append(action)
        self._actions.sort(key=lambda row: row.occurred_at, reverse=True)

    async def list_operator_actions(
        self,
        *,
        action_type: str | None = None,
        limit: int = 200,
    ) -> list[OperatorAction]:
        rows = [
            row for row in self._actions if action_type is None or row.action_type == action_type
        ]
        return rows[:limit]

    async def save_runbook_evidence(self, evidence: RunbookEvidence) -> None:
        self._runbook = [
            existing for existing in self._runbook if existing.evidence_id != evidence.evidence_id
        ]
        self._runbook.append(evidence)

    async def save_alert_event(self, event: AlertEvent) -> None:
        self._alerts = [
            existing for existing in self._alerts if existing.alert_id != event.alert_id
        ]
        self._alerts.append(event)

    async def save_readiness_snapshot(self, snapshot: ReadinessSnapshot) -> None:
        self._readiness = [
            existing for existing in self._readiness if existing.snapshot_id != snapshot.snapshot_id
        ]
        self._readiness.append(snapshot)
        self._readiness.sort(key=lambda row: row.generated_at, reverse=True)

    async def latest_readiness_snapshot(self, profile: str) -> ReadinessSnapshot | None:
        for row in self._readiness:
            if row.profile.value == profile:
                return row
        return None

    async def save_api_key(self, key: OperatorApiKey) -> None:
        self._api_keys[key.key_id] = key

    async def get_api_key_by_hash(self, key_hash: str) -> OperatorApiKey | None:
        for key in self._api_keys.values():
            if key.key_hash == key_hash:
                return key
        return None

    async def revoke_api_key(
        self,
        key_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> None:
        key = self._api_keys.get(key_id)
        if key is None:
            return
        self._api_keys[key_id] = OperatorApiKey(
            key_id=key.key_id,
            key_hash=key.key_hash,
            role=key.role,
            created_at=key.created_at,
            created_by=key.created_by,
            revoked_at=revoked_at,
            label=key.label,
        )
