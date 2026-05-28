"""Write-side operations for the Postgres model registry."""

from __future__ import annotations

import json
import socket
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from quant_platform.infrastructure.postgres.retry import with_retry as _with_retry
from quant_platform.services.research_service.modeling.registry.model_registry import (
    RegisteredModel,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresModelRegistryMutationsMixin:
    """Mutation methods for registered model rows."""

    _engine: AsyncEngine

    if TYPE_CHECKING:

        async def get_active_model(self, strategy_name: str) -> RegisteredModel | None: ...

    async def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModel:
        """Insert a new model row and flip any prior active rows inactive."""
        model_id = uuid.uuid4()
        meta_json = json.dumps(metadata or {}, default=str)
        activated_by = _hostname()

        async def _do() -> None:
            async with self._engine.begin() as conn:
                prev_row = (
                    (
                        await conn.execute(
                            text(
                                """
                                SELECT model_id FROM registered_models
                                WHERE strategy_name = :strategy_name AND active = true
                                LIMIT 1
                                """
                            ),
                            {"strategy_name": strategy_name},
                        )
                    )
                    .mappings()
                    .first()
                )
                from_model_id = uuid.UUID(str(prev_row["model_id"])) if prev_row else None
                await conn.execute(
                    text(
                        """
                        UPDATE registered_models
                        SET active = false
                        WHERE strategy_name = :strategy_name AND active = true
                        """
                    ),
                    {"strategy_name": strategy_name},
                )
                await conn.execute(
                    text(
                        """
                        INSERT INTO registered_models
                            (model_id, strategy_name, model_version, feature_set_version,
                             created_at, active, metadata_json)
                        VALUES
                            (:model_id, :strategy_name, :model_version, :feature_set_version,
                             :created_at, true, CAST(:metadata_json AS jsonb))
                        """
                    ),
                    {
                        "model_id": model_id,
                        "strategy_name": strategy_name,
                        "model_version": model_version,
                        "feature_set_version": feature_set_version,
                        "created_at": as_of,
                        "metadata_json": meta_json,
                    },
                )
                await conn.execute(
                    text(
                        """
                        INSERT INTO model_activation_audit
                            (audit_id, strategy_name, to_model_id, from_model_id,
                             activated_at, activated_by)
                        VALUES
                            (:audit_id, :strategy_name, :to_model_id, :from_model_id,
                             :activated_at, :activated_by)
                        """
                    ),
                    {
                        "audit_id": uuid.uuid4(),
                        "strategy_name": strategy_name,
                        "to_model_id": model_id,
                        "from_model_id": from_model_id,
                        "activated_at": as_of,
                        "activated_by": activated_by,
                    },
                )

        await _with_retry(_do)
        return RegisteredModel(
            model_id=model_id,
            strategy_name=strategy_name,
            model_version=model_version,
            feature_set_version=feature_set_version,
            created_at=as_of,
            metadata=dict(metadata or {}),
            active=True,
        )

    async def retire_model(self, strategy_name: str) -> int:
        """Flip the currently-active model row to ``active = false``."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    UPDATE registered_models
                    SET active = false
                    WHERE strategy_name = :strategy_name AND active = true
                    """
                ),
                {"strategy_name": strategy_name},
            )
        return result.rowcount if result.rowcount is not None else 0

    async def rollback_to_version(
        self,
        strategy_name: str,
        target_version: str,
    ) -> RegisteredModel:
        """Atomically restore ``target_version`` as the active row."""
        async with self._engine.begin() as conn:
            target_row = (
                (
                    await conn.execute(
                        text(
                            """
                            SELECT model_id
                            FROM registered_models
                            WHERE strategy_name = :strategy_name
                              AND model_version = :model_version
                            """
                        ),
                        {
                            "strategy_name": strategy_name,
                            "model_version": target_version,
                        },
                    )
                )
                .mappings()
                .first()
            )
            if target_row is None:
                raise LookupError(
                    f"rollback target not found: strategy={strategy_name!r} "
                    f"version={target_version!r}"
                )
            await conn.execute(
                text(
                    """
                    UPDATE registered_models
                    SET active = false
                    WHERE strategy_name = :strategy_name AND active = true
                    """
                ),
                {"strategy_name": strategy_name},
            )
            await conn.execute(
                text(
                    """
                    UPDATE registered_models
                    SET active = true
                    WHERE model_id = :model_id
                    """
                ),
                {"model_id": target_row["model_id"]},
            )
        active = await self.get_active_model(strategy_name)
        if active is None:
            raise RuntimeError(
                f"rollback completed but no active model found for strategy {strategy_name!r}"
            )
        return active


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"
