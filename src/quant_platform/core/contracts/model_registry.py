"""Model registry contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


@runtime_checkable
class RegisteredModelRecord(Protocol):
    """Structural view of a registered research/model artifact."""

    model_id: uuid.UUID
    strategy_name: str
    model_version: str
    feature_set_version: str
    created_at: datetime
    metadata: dict[str, object]
    active: bool


@runtime_checkable
class ModelRegistryRepository(Protocol):
    """Persistence for promoted model versions."""

    async def list_models(
        self,
        strategy_name: str | None = None,
    ) -> list[RegisteredModelRecord]: ...

    async def get_model(
        self,
        strategy_name: str,
        model_version: str,
    ) -> RegisteredModelRecord | None: ...

    async def get_active_model(self, strategy_name: str) -> RegisteredModelRecord | None: ...

    async def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModelRecord: ...

    async def retire_model(self, strategy_name: str) -> int: ...

    async def rollback_to_version(
        self,
        strategy_name: str,
        target_version: str,
    ) -> RegisteredModelRecord: ...
