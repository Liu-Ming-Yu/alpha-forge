"""Model-registry wiring used during engine initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from quant_platform.engines.runtime.registry import check_model_staleness, maybe_await

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable
    from datetime import datetime

    from quant_platform.services.research_service.modeling.registry.model_registry import (
        FeatureJob,
        RegisteredModel,
    )


class EngineModelRegistry(Protocol):
    def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModel | Awaitable[RegisteredModel]: ...

    def schedule_feature_job(
        self,
        *,
        model_id: uuid.UUID,
        strategy_name: str,
        feature_set_version: str,
        interval_seconds: float,
        as_of: datetime,
    ) -> FeatureJob | Awaitable[FeatureJob]: ...


async def register_engine_model_and_schedule_job(
    model_registry: EngineModelRegistry,
    *,
    engine_name: str,
    engine_version: str,
    feature_set_version: str,
    run_mode: object,
    max_positions: int,
    interval_seconds: float,
    as_of: datetime,
    max_model_age_hours: float,
) -> FeatureJob:
    """Register the engine model and schedule its recurring feature job."""
    run_mode_value = getattr(run_mode, "value", run_mode)
    model = cast(
        "RegisteredModel",
        await maybe_await(
            model_registry.register_model(
                strategy_name=engine_name,
                model_version=engine_version,
                feature_set_version=feature_set_version,
                as_of=as_of,
                metadata={
                    "run_mode": str(run_mode_value),
                    "max_positions": max_positions,
                },
            )
        ),
    )
    await check_model_staleness(
        model,
        as_of=as_of,
        max_age_hours=max_model_age_hours,
    )
    scheduled_job = await maybe_await(
        model_registry.schedule_feature_job(
            model_id=model.model_id,
            strategy_name=engine_name,
            feature_set_version=model.feature_set_version,
            interval_seconds=interval_seconds,
            as_of=as_of,
        )
    )
    return cast("FeatureJob", scheduled_job)
