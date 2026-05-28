"""PostgreSQL-backed model registry adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import structlog

from quant_platform.infrastructure.postgres.feature_jobs import PostgresFeatureJobsMixin
from quant_platform.infrastructure.postgres.model_registry_mutations import (
    PostgresModelRegistryMutationsMixin,
)
from quant_platform.infrastructure.postgres.model_registry_queries import (
    PostgresModelRegistryQueriesMixin,
)
from quant_platform.services.research_service.modeling.registry.model_registry import (
    InMemoryModelRegistry,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)


class PostgresModelRegistry(
    PostgresModelRegistryMutationsMixin,
    PostgresModelRegistryQueriesMixin,
    PostgresFeatureJobsMixin,
):
    """Async, Postgres-backed model registry adapter."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine


@overload
def build_model_registry(dsn: None) -> InMemoryModelRegistry: ...


@overload
def build_model_registry(dsn: str) -> PostgresModelRegistry: ...


@overload
def build_model_registry(dsn: str | None) -> InMemoryModelRegistry | PostgresModelRegistry: ...


def build_model_registry(dsn: str | None) -> InMemoryModelRegistry | PostgresModelRegistry:
    """Select Postgres when a DSN is configured, in-memory otherwise."""
    if not dsn:
        log.info("model_registry.backend", backend="in_memory")
        return InMemoryModelRegistry()
    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    engine = create_pg_engine(dsn)
    log.info("model_registry.backend", backend="postgres")
    return PostgresModelRegistry(engine)


__all__ = ["PostgresModelRegistry", "build_model_registry"]
