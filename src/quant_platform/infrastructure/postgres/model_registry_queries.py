"""Read-side operations for the Postgres model registry."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.postgres.model_registry_rows import (
    row_to_registered_model,
)
from quant_platform.infrastructure.postgres.retry import with_retry as _with_retry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from quant_platform.services.research_service.modeling.registry.model_registry import (
        RegisteredModel,
    )


class PostgresModelRegistryQueriesMixin:
    """Query methods for registered model rows."""

    _engine: AsyncEngine

    async def get_active_model(self, strategy_name: str) -> RegisteredModel | None:
        async def _do() -> RegisteredModel | None:
            async with self._engine.connect() as conn:
                row = (
                    (
                        await conn.execute(
                            text(
                                """
                                SELECT model_id, strategy_name, model_version,
                                       feature_set_version, created_at, metadata_json, active
                                FROM registered_models
                                WHERE strategy_name = :strategy_name AND active = true
                                ORDER BY created_at DESC
                                LIMIT 1
                                """
                            ),
                            {"strategy_name": strategy_name},
                        )
                    )
                    .mappings()
                    .first()
                )
            return None if row is None else row_to_registered_model(dict(row))

        return await _with_retry(_do)

    async def list_models(self, strategy_name: str | None = None) -> list[RegisteredModel]:
        """List registered models, most-recent-first."""
        if strategy_name:
            query = text(
                """
                SELECT model_id, strategy_name, model_version,
                       feature_set_version, created_at, metadata_json, active
                FROM registered_models
                WHERE strategy_name = :strategy_name
                ORDER BY created_at DESC
                """
            )
            params: dict[str, object] = {"strategy_name": strategy_name}
        else:
            query = text(
                """
                SELECT model_id, strategy_name, model_version,
                       feature_set_version, created_at, metadata_json, active
                FROM registered_models
                ORDER BY strategy_name ASC, created_at DESC
                """
            )
            params = {}
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query, params)).mappings().all()
        return [row_to_registered_model(dict(row)) for row in rows]

    async def get_model(
        self,
        strategy_name: str,
        model_version: str,
    ) -> RegisteredModel | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            """
                            SELECT model_id, strategy_name, model_version,
                                   feature_set_version, created_at, metadata_json, active
                            FROM registered_models
                            WHERE strategy_name = :strategy_name
                              AND model_version = :model_version
                            """
                        ),
                        {"strategy_name": strategy_name, "model_version": model_version},
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else row_to_registered_model(dict(row))

    async def get_model_age_hours(self, strategy_name: str) -> float | None:
        """Return hours since active model registration, or None if absent."""
        model = await self.get_active_model(strategy_name)
        if model is None:
            return None
        created = model.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (datetime.now(tz=UTC) - created).total_seconds() / 3600.0
