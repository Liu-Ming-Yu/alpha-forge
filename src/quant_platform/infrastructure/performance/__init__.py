"""Compatibility exports for performance repository adapters."""

from __future__ import annotations

from quant_platform.infrastructure.performance.inmemory import InMemoryPerformanceRepository
from quant_platform.infrastructure.performance.postgres import PostgresPerformanceRepository
from quant_platform.infrastructure.performance.status import (
    build_performance_report,
    build_shadow_paper_parity_status,
    build_signal_gate_status,
    build_text_gate_status,
)

__all__ = [
    "InMemoryPerformanceRepository",
    "PostgresPerformanceRepository",
    "build_performance_repository",
    "build_performance_report",
    "build_shadow_paper_parity_status",
    "build_signal_gate_status",
    "build_text_gate_status",
]


def build_performance_repository(
    dsn: str | None,
) -> InMemoryPerformanceRepository | PostgresPerformanceRepository:
    """Select Postgres when a DSN is configured, otherwise in-memory."""
    if not dsn:
        return InMemoryPerformanceRepository()
    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    return PostgresPerformanceRepository(create_pg_engine(dsn))
