"""Migration and schema verification composition helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.infrastructure.support.migrations import (
    migrate_database,
    packaged_head,
    validate_packaged_migration_chain,
    verify_alembic_head,
    verify_postgres_schema_if_configured,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def migrate_database_to_head(settings: PlatformSettings) -> str:
    return migrate_database(settings)


def validate_packaged_migrations() -> str:
    return validate_packaged_migration_chain()


def alembic_packaged_head() -> str:
    return packaged_head()


async def verify_database_head(dsn: str) -> None:
    await verify_alembic_head(dsn)


async def verify_postgres_schema(settings: PlatformSettings) -> None:
    await verify_postgres_schema_if_configured(settings)
