"""Repository adapters for alpha forecast materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.postgres.feature_repository import PostgresFeatureRepository
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import FeatureRepository


def build_alpha_forecast_feature_repository(settings: PlatformSettings) -> FeatureRepository:
    """Build only the feature repository needed for read-only forecast evidence."""
    if not settings.storage.postgres_dsn:
        return InMemoryFeatureRepository()

    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    pg_engine = create_pg_engine(
        settings.storage.postgres_dsn,
        pool_size=settings.storage.postgres_pool_min,
        max_overflow=max(
            0,
            settings.storage.postgres_pool_max - settings.storage.postgres_pool_min,
        ),
    )
    return PostgresFeatureRepository(pg_engine)


__all__ = ["build_alpha_forecast_feature_repository"]
