"""Infrastructure operator adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_platform.application.operator.requests import NoInputRequest
    from quant_platform.config import PlatformSettings


class InfraAdapters:
    """Concrete infrastructure adapters backed by migration helpers."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    def migrate(self, _request: NoInputRequest) -> str:
        from quant_platform.bootstrap.persistence.migrations import migrate_database_to_head

        return migrate_database_to_head(self._settings)

    def migrations_check(self, _request: NoInputRequest) -> str:
        from quant_platform.bootstrap.persistence.migrations import validate_packaged_migrations

        return validate_packaged_migrations()

    async def verify_schema(self, _request: NoInputRequest) -> None:
        from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema

        await verify_postgres_schema(self._settings)


__all__ = ["InfraAdapters"]
