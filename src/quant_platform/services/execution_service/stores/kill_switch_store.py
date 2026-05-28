"""Durable kill-switch state store.

Retires the process-local kill switch exposed by
:class:`OrderThrottle` prior to the Validation & Governance sprint: a
process restart used to silently clear the switch, so a cash-drift halt
would persist only for the lifetime of the worker that observed the
drift.  With this module the state lives in Postgres
(``kill_switch_state`` table, Alembic revision ``004``) and every
session hydrates it at startup.

The two concrete stores follow the in-memory / Postgres split used
elsewhere in the platform:

- :class:`InMemoryKillSwitchStore`: unit-test double; state lives inside
  the store instance.
- :class:`PostgresKillSwitchStore`: production default when
  ``QP__STORAGE__POSTGRES_DSN`` is set; hydrates and persists against
  the singleton row.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import structlog
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class KillSwitchState:
    """Read model for the singleton kill-switch row."""

    active: bool
    reason: str | None
    activated_at: datetime | None
    activated_by: str | None
    cleared_at: datetime | None = None


class KillSwitchStore(Protocol):
    """Protocol for reading / writing the durable kill-switch state."""

    async def get(self) -> KillSwitchState: ...
    async def activate(self, *, reason: str, activated_by: str, as_of: datetime) -> None: ...
    async def clear(self, *, operator_id: str, as_of: datetime) -> None: ...
    async def health_check(self) -> bool: ...


class InMemoryKillSwitchStore:
    """Test double that keeps kill-switch state inside the instance."""

    def __init__(self) -> None:
        self._state = KillSwitchState(
            active=False, reason=None, activated_at=None, activated_by=None
        )
        self._lock = asyncio.Lock()

    async def get(self) -> KillSwitchState:
        return self._state

    async def activate(self, *, reason: str, activated_by: str, as_of: datetime) -> None:
        async with self._lock:
            self._state = KillSwitchState(
                active=True,
                reason=reason,
                activated_at=as_of,
                activated_by=activated_by,
            )

    async def clear(self, *, operator_id: str, as_of: datetime) -> None:
        async with self._lock:
            self._state = KillSwitchState(
                active=False,
                reason=None,
                activated_at=None,
                activated_by=operator_id,
                cleared_at=as_of,
            )

    async def health_check(self) -> bool:
        return True


class PostgresKillSwitchStore:
    """Singleton-row store backed by ``kill_switch_state``.

    Uses ``INSERT ... ON CONFLICT DO UPDATE`` to stay compatible with a
    fresh database that somehow skipped the migration's seeding step.
    """

    _ID = "default"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get(self) -> KillSwitchState:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            """
                        SELECT active, reason, activated_at, activated_by, cleared_at
                        FROM kill_switch_state
                        WHERE id = :id
                        """
                        ),
                        {"id": self._ID},
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return KillSwitchState(active=False, reason=None, activated_at=None, activated_by=None)
        return KillSwitchState(
            active=bool(row["active"]),
            reason=row["reason"],
            activated_at=row["activated_at"],
            activated_by=row["activated_by"],
            cleared_at=row.get("cleared_at"),
        )

    async def health_check(self) -> bool:
        """Verify DB connectivity by executing SELECT 1."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            log.critical(
                "kill_switch_store.health_check.failed",
                detail="Postgres kill-switch store is unreachable; cannot guarantee durable state",
            )
            return False

    async def activate(self, *, reason: str, activated_by: str, as_of: datetime) -> None:
        await self._upsert(
            active=True,
            reason=reason,
            activated_at=as_of,
            activated_by=activated_by,
        )

    async def clear(self, *, operator_id: str, as_of: datetime) -> None:
        await self._upsert(
            active=False,
            reason=None,
            activated_at=None,
            activated_by=operator_id,
            cleared_at=as_of,
        )

    async def _upsert(
        self,
        *,
        active: bool,
        reason: str | None,
        activated_at: datetime | None,
        activated_by: str | None,
        cleared_at: datetime | None = None,
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO kill_switch_state
                        (
                            id,
                            active,
                            reason,
                            activated_at,
                            activated_by,
                            cleared_at,
                            updated_at
                        )
                    VALUES
                        (
                            :id,
                            :active,
                            :reason,
                            :activated_at,
                            :activated_by,
                            :cleared_at,
                            :updated_at
                        )
                    ON CONFLICT (id) DO UPDATE SET
                        active = EXCLUDED.active,
                        reason = EXCLUDED.reason,
                        activated_at = EXCLUDED.activated_at,
                        activated_by = EXCLUDED.activated_by,
                        cleared_at = EXCLUDED.cleared_at,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "id": self._ID,
                    "active": active,
                    "reason": reason,
                    "activated_at": activated_at,
                    "activated_by": activated_by,
                    "cleared_at": cleared_at,
                    "updated_at": datetime.now(tz=UTC),
                },
            )


async def validate_kill_switch_store(store: KillSwitchStore) -> None:
    """Verify the kill-switch store is reachable.

    Raises RuntimeError if the store fails its health check.  Call this from
    async session startup after composing the store so a Postgres connectivity
    failure is a loud crash, not a silent fallback.
    """
    if not await store.health_check():
        raise RuntimeError(
            "kill_switch_store health check failed; refusing to start session "
            "without a reachable durable kill-switch store"
        )
