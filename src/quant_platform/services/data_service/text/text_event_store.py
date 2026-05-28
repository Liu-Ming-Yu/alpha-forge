"""Text event store implementations (Phase 5).

Provides ``InMemoryTextEventStore`` for dev/CI/backtest and
``PostgresTextEventStore`` for production.

Both implement the ``TextEventProvider`` protocol defined in
``core/contracts/text_data.py``.

Wiring:
    ``create_paper_session()`` and ``create_live_session()`` in ``session.py``
    select ``PostgresTextEventStore`` when ``QP__STORAGE__POSTGRES_DSN`` is set,
    otherwise fall back to ``InMemoryTextEventStore``.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import text

from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)


class InMemoryTextEventStore:
    """Write-once, in-process text event store.

    Suitable for dev, CI, and paper trading sessions without Postgres.
    All data is lost on process restart.

    Invariants:
    - store_event() is idempotent on event_id.
    - get_events() returns events sorted by occurred_at ascending.
    """

    def __init__(self) -> None:
        self._events: dict[uuid.UUID, TextEvent] = {}
        # Secondary index: (instrument_id | None) → list[event_id]
        self._by_instrument: dict[uuid.UUID | None, list[uuid.UUID]] = defaultdict(list)
        log.info("text_event_store.backend", backend="in_memory")

    # ------------------------------------------------------------------
    # TextEventProvider protocol
    # ------------------------------------------------------------------

    async def store_event(self, event: TextEvent) -> None:
        """Persist a text event.  No-op if event_id already exists."""
        if event.event_id in self._events:
            return
        self._events[event.event_id] = event
        self._by_instrument[event.instrument_id].append(event.event_id)

    async def get_events(
        self,
        start: datetime,
        end: datetime,
        *,
        instrument_ids: list[uuid.UUID] | None = None,
        event_types: list[TextEventType] | None = None,
    ) -> list[TextEvent]:
        """Return events in [start, end) sorted by occurred_at."""
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be UTC-aware")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        results: list[TextEvent] = []
        for event in self._events.values():
            if not (start <= event.occurred_at < end):
                continue
            if event_types is not None and event.event_type not in event_types:
                continue
            if (
                instrument_ids is not None
                and len(instrument_ids) > 0
                and event.instrument_id not in instrument_ids
            ):
                # Macro events (instrument_id=None) are excluded when a
                # specific instrument filter is provided.
                continue
            results.append(event)

        results.sort(key=lambda e: e.occurred_at)
        return results


class PostgresTextEventStore:
    """PostgreSQL-backed text event store.

    Table: text_events (created by Alembic revision 006).
    Schema:
        id UUID PK
        instrument_id UUID (nullable)
        event_type TEXT
        occurred_at TIMESTAMPTZ
        source_uri TEXT
        artifact_uri TEXT
        metadata JSONB
        created_at TIMESTAMPTZ DEFAULT now()

    Invariants:
    - INSERT ... ON CONFLICT DO NOTHING ensures idempotency on event_id.
    - get_events() adds a BETWEEN filter on occurred_at and an optional
      instrument_id IN clause.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """Args:
        engine: Shared async SQLAlchemy engine used by the other Postgres
            repositories in the session.
        """
        self._engine = engine
        log.info("text_event_store.backend", backend="postgres")

    async def store_event(self, event: TextEvent) -> None:
        """Insert a text event row; idempotent via ON CONFLICT DO NOTHING."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO text_events
                        (id, instrument_id, event_type, occurred_at,
                         source_uri, artifact_uri, metadata, provider, dedupe_key,
                         content_hash, ingestion_status, source_published_at)
                    VALUES
                        (:id, :instrument_id, :event_type, :occurred_at,
                         :source_uri, :artifact_uri, CAST(:metadata AS JSONB),
                         :provider, :dedupe_key, :content_hash, :ingestion_status,
                         :source_published_at)
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": event.event_id,
                    "instrument_id": event.instrument_id,
                    "event_type": event.event_type.value,
                    "occurred_at": event.occurred_at,
                    "source_uri": event.source_uri,
                    "artifact_uri": event.artifact_uri,
                    "metadata": json.dumps(dict(event.metadata)),
                    "provider": event.metadata.get("provider"),
                    "dedupe_key": event.metadata.get("dedupe_key"),
                    "content_hash": event.metadata.get("content_hash"),
                    "ingestion_status": event.metadata.get("ingestion_status"),
                    "source_published_at": event.metadata.get("source_published_at"),
                },
            )

    async def get_events(
        self,
        start: datetime,
        end: datetime,
        *,
        instrument_ids: list[uuid.UUID] | None = None,
        event_types: list[TextEventType] | None = None,
    ) -> list[TextEvent]:
        """Query text events from Postgres."""
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be UTC-aware")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        params: dict[str, object] = {"start": start, "end": end}
        clauses = ["occurred_at >= :start", "occurred_at < :end"]

        # Use = ANY(:param) with a list parameter rather than building a dynamic
        # IN (...) clause, so the SQL template never contains f-string interpolation.
        if instrument_ids is not None and len(instrument_ids) > 0:
            clauses.append("instrument_id = ANY(:instrument_ids)")
            params["instrument_ids"] = instrument_ids

        if event_types is not None and len(event_types) > 0:
            clauses.append("event_type = ANY(:event_types)")
            params["event_types"] = [et.value for et in event_types]

        base_sql = (
            "SELECT id, instrument_id, event_type, occurred_at, source_uri, artifact_uri, metadata "
            "FROM text_events WHERE " + " AND ".join(clauses) + " ORDER BY occurred_at ASC"
        )

        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(base_sql), params)).mappings().all()

        events: list[TextEvent] = []
        for row in rows:
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            events.append(
                TextEvent(
                    event_id=uuid.UUID(str(row["id"])),
                    instrument_id=uuid.UUID(str(row["instrument_id"]))
                    if row["instrument_id"]
                    else None,
                    event_type=TextEventType(str(row["event_type"])),
                    occurred_at=row["occurred_at"],
                    source_uri=str(row["source_uri"]),
                    artifact_uri=str(row["artifact_uri"]),
                    metadata=meta,
                )
            )
        return events
