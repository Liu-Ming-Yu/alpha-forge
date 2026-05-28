"""PostgreSQL-backed repository adapter exports.

These adapters satisfy the same ``OrderRepository``, ``PositionRepository``,
and ``AuditSink`` protocols as their in-memory counterparts, backed by
SQLAlchemy async engine + raw SQL (no ORM models; keeps the domain layer
independent of persistence schema).

The public factory ``create_pg_engine()`` accepts a DSN string and returns
an ``AsyncEngine``.  Session wiring in ``session.py`` uses it to decide
in-memory vs Postgres at startup: if ``StorageSettings.postgres_dsn`` is
non-empty, Postgres adapters are used; otherwise in-memory.

Schema migrations are managed by Alembic. The concrete adapters live in
focused modules; this package module re-exports the durable adapter surface.
"""

from __future__ import annotations

from quant_platform.infrastructure.postgres.audit_sink import PostgresAuditSink
from quant_platform.infrastructure.postgres.order_repository import PostgresOrderRepository
from quant_platform.infrastructure.postgres.position_repository import PostgresPositionRepository
from quant_platform.infrastructure.postgres.schema_history import BOOTSTRAP_SQL
from quant_platform.infrastructure.postgres.support import create_pg_engine

__all__ = [
    "BOOTSTRAP_SQL",
    "PostgresAuditSink",
    "PostgresOrderRepository",
    "PostgresPositionRepository",
    "create_pg_engine",
]
