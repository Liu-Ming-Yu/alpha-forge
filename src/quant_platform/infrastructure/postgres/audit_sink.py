"""PostgreSQL audit sink adapter."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.postgres.support import retry_transient

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.core.events import DomainEvent


class PostgresAuditSink:
    """PostgreSQL-backed append-only audit log."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @retry_transient()
    async def record(self, event: DomainEvent, context: dict[str, object]) -> None:
        payload = {
            k: str(v) if isinstance(v, (uuid.UUID, datetime, Decimal)) else v
            for k, v in vars(event).items()
        }
        ctx_safe = {
            k: str(v) if isinstance(v, (uuid.UUID, datetime, Decimal)) else v
            for k, v in context.items()
        }
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO audit_log (event_type, event_payload, context)
                    VALUES (:event_type, :payload, :context)
                """),
                {
                    "event_type": type(event).__name__,
                    "payload": json.dumps(payload, default=str),
                    "context": json.dumps(ctx_safe, default=str),
                },
            )

    @retry_transient()
    async def list_events(
        self,
        *,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        """Return audit rows sorted by recorded_at descending."""
        if limit < 1:
            limit = 1
        if limit > 1000:
            limit = 1000
        if offset < 0:
            offset = 0

        where_parts: list[str] = []
        params: dict[str, object] = {"limit": limit, "offset": offset}
        if event_type:
            where_parts.append("event_type = :event_type")
            params["event_type"] = event_type
        if since is not None:
            where_parts.append("recorded_at >= :since")
            params["since"] = since
        if until is not None:
            where_parts.append("recorded_at < :until")
            params["until"] = until

        where_clause = ("WHERE " + " AND ".join(where_parts) + " ") if where_parts else ""
        sql = text(
            "SELECT entry_id AS audit_id, event_type, event_payload, context, recorded_at "
            "FROM audit_log " + where_clause + "ORDER BY recorded_at DESC, entry_id DESC "
            "LIMIT :limit OFFSET :offset"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()
        out: list[dict[str, object]] = []
        for row in rows:
            payload_raw = row["event_payload"]
            ctx_raw = row["context"]
            out.append(
                {
                    "audit_id": str(row["audit_id"]),
                    "event_type": row["event_type"],
                    "event_payload": (
                        json.loads(payload_raw)
                        if isinstance(payload_raw, (str, bytes))
                        else payload_raw
                    ),
                    "context": (
                        json.loads(ctx_raw) if isinstance(ctx_raw, (str, bytes)) else ctx_raw
                    ),
                    "recorded_at": str(row["recorded_at"]),
                }
            )
        return out
