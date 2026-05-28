"""Durable completed-order hint stores."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)


class CompletedOrderHintStore(Protocol):
    """Remembers which orders already received a BrokerOrderCompleted."""

    async def list_all(self, run_id: uuid.UUID | None = None) -> set[uuid.UUID]: ...

    async def add(self, order_id: uuid.UUID, *, run_id: uuid.UUID) -> None: ...

    async def remove(self, order_id: uuid.UUID) -> None: ...


class InMemoryCompletedOrderHintStore:
    """Process-local completed-order hint store."""

    def __init__(self) -> None:
        self._rows: dict[uuid.UUID, uuid.UUID | None] = {}

    async def list_all(self, run_id: uuid.UUID | None = None) -> set[uuid.UUID]:
        if run_id is None:
            return set(self._rows.keys())
        return {oid for oid, stored_run in self._rows.items() if stored_run == run_id}

    async def add(self, order_id: uuid.UUID, *, run_id: uuid.UUID) -> None:
        self._rows[order_id] = run_id

    async def remove(self, order_id: uuid.UUID) -> None:
        self._rows.pop(order_id, None)


class PostgresCompletedOrderHintStore:
    """Postgres-backed completed-order hint store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def list_all(self, run_id: uuid.UUID | None = None) -> set[uuid.UUID]:
        from sqlalchemy import text

        query = "SELECT order_id FROM completed_order_hints"
        params: dict[str, object] = {}
        if run_id is not None:
            query += " WHERE run_id = :run_id"
            params["run_id"] = str(run_id)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(query), params)).mappings().all()

        out: set[uuid.UUID] = set()
        for row in rows:
            try:
                out.add(uuid.UUID(str(row["order_id"])))
            except Exception as exc:  # pragma: no cover - defensive
                log.debug(
                    "completed_order_hint.invalid_order_id",
                    raw_order_id=str(row.get("order_id")),
                    error=str(exc),
                )
                continue
        return out

    async def add(self, order_id: uuid.UUID, *, run_id: uuid.UUID) -> None:
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO completed_order_hints (order_id, run_id, completed_at)
                    VALUES (:order_id, :run_id, :completed_at)
                    ON CONFLICT (order_id) DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        completed_at = EXCLUDED.completed_at
                    """
                ),
                {
                    "order_id": str(order_id),
                    "run_id": str(run_id),
                    "completed_at": datetime.now(tz=UTC),
                },
            )

    async def remove(self, order_id: uuid.UUID) -> None:
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM completed_order_hints WHERE order_id = :order_id"),
                {"order_id": str(order_id)},
            )
