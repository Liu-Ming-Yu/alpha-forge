"""Execution-store composition helpers."""

from __future__ import annotations

import structlog

from quant_platform.infrastructure.postgres.repositories import create_pg_engine
from quant_platform.services.execution_service.stores.kill_switch_store import (
    InMemoryKillSwitchStore,
    KillSwitchStore,
    PostgresKillSwitchStore,
)
from quant_platform.services.execution_service.stores.pending_settlement_store import (
    CompletedOrderHintStore,
    InMemoryCompletedOrderHintStore,
    InMemoryPendingSettlementStore,
    PendingSettlementStore,
    PostgresCompletedOrderHintStore,
    PostgresPendingSettlementStore,
)

log = structlog.get_logger(__name__)


def build_kill_switch_store(dsn: str | None) -> KillSwitchStore:
    """Select Postgres when ``dsn`` is configured, in-memory otherwise."""
    if not dsn:
        log.info("kill_switch_store.backend", backend="in_memory")
        return InMemoryKillSwitchStore()
    engine = create_pg_engine(dsn)
    log.info("kill_switch_store.backend", backend="postgres")
    return PostgresKillSwitchStore(engine)


def build_pending_settlement_store(dsn: str | None) -> PendingSettlementStore:
    """Select Postgres when ``dsn`` is configured, in-memory otherwise."""
    if not dsn:
        log.info("pending_settlement_store.backend", backend="in_memory")
        return InMemoryPendingSettlementStore()
    engine = create_pg_engine(dsn)
    log.info("pending_settlement_store.backend", backend="postgres")
    return PostgresPendingSettlementStore(engine)


def build_completed_order_hint_store(dsn: str | None) -> CompletedOrderHintStore:
    """Select Postgres when ``dsn`` is configured, in-memory otherwise."""
    if not dsn:
        log.info("completed_order_hint_store.backend", backend="in_memory")
        return InMemoryCompletedOrderHintStore()
    engine = create_pg_engine(dsn)
    log.info("completed_order_hint_store.backend", backend="postgres")
    return PostgresCompletedOrderHintStore(engine)
