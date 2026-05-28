"""Operator evidence and authentication domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


@dataclass(frozen=True)
class AlphaReadinessEvidence:
    """Operator-facing readiness state for promoted alpha sources."""

    signal_name: str
    signal_type: str
    promotion_state: str
    ramp_level: Decimal
    artifact_hash_verified: bool
    feature_schema_verified: bool
    source_fresh: bool
    rollback_target: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.promotion_state in {"paper", "live"}
            and self.artifact_hash_verified
            and self.feature_schema_verified
            and self.source_fresh
            and self.ramp_level >= Decimal("0")
        )


@dataclass(frozen=True)
class OperatorAction:
    """Durable human/operator command evidence."""

    action_id: uuid.UUID
    occurred_at: datetime
    action_type: str
    actor: str
    reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if not self.action_type.strip():
            raise ValueError("action_type must not be empty")
        if not self.actor.strip():
            raise ValueError("actor must not be empty")


@dataclass(frozen=True)
class RunbookEvidence:
    """Persisted evidence that a production runbook was executed."""

    evidence_id: uuid.UUID
    runbook_name: str
    executed_at: datetime
    actor: str
    status: str
    artifact_uri: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.executed_at.tzinfo is None:
            raise ValueError("executed_at must be timezone-aware")
        if not self.runbook_name.strip():
            raise ValueError("runbook_name must not be empty")
        if not self.actor.strip():
            raise ValueError("actor must not be empty")


@dataclass(frozen=True)
class AlertEvent:
    """Persisted operational alert for SLO and incident review."""

    alert_id: uuid.UUID
    occurred_at: datetime
    severity: str
    component: str
    message: str
    resolved_at: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.resolved_at is not None and self.resolved_at.tzinfo is None:
            raise ValueError("resolved_at must be timezone-aware")
        if self.resolved_at is not None and self.resolved_at < self.occurred_at:
            raise ValueError("resolved_at must be >= occurred_at")
        if self.severity not in {"info", "warning", "error", "critical"}:
            raise ValueError("severity must be one of info/warning/error/critical")


@dataclass(frozen=True)
class OperatorApiKey:
    """Hashed operator API credential with role and revocation state."""

    key_id: uuid.UUID
    key_hash: str
    role: str
    created_at: datetime
    created_by: str
    revoked_at: datetime | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("revoked_at must be timezone-aware")
        if self.revoked_at is not None and self.revoked_at < self.created_at:
            raise ValueError("revoked_at must be >= created_at")
        if self.role not in {"viewer", "operator", "admin"}:
            raise ValueError("role must be viewer/operator/admin")
        if not self.key_hash.strip():
            raise ValueError("key_hash must not be empty")
        if not self.created_by.strip():
            raise ValueError("created_by must not be empty")
