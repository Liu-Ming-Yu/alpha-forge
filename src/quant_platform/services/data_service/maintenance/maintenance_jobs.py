"""Feature-job execution helpers for data maintenance supervision."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Sequence
    from datetime import datetime

    from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
        DataMaintenanceResult,
        DataMaintenanceScheduler,
    )

log = structlog.get_logger(__name__)

T = TypeVar("T")


class FeatureJobRegistry(Protocol):
    def due_feature_jobs(
        self,
        as_of: datetime,
    ) -> Sequence[FeatureJobRef] | Awaitable[Sequence[FeatureJobRef]]: ...

    def mark_job_completed(
        self,
        job_id: uuid.UUID,
        *,
        run_at: datetime,
        success: bool,
    ) -> object | Awaitable[object]: ...


class FeatureJobRef(Protocol):
    """Minimal feature-job shape consumed by data maintenance."""

    @property
    def job_id(self) -> uuid.UUID: ...

    @property
    def feature_set_version(self) -> str: ...


@dataclass
class MaintenanceTick:
    """Result summary for a single supervisor iteration."""

    as_of: datetime
    jobs_processed: int
    features_stored: int
    errors: int


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Await async registry adapters while allowing sync in-memory fakes."""
    if inspect.isawaitable(value):
        return await value
    return value


async def run_due_feature_jobs(
    *,
    model_registry: object,
    scheduler: DataMaintenanceScheduler,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
) -> MaintenanceTick:
    """Run all feature jobs due at ``as_of`` and record completion state."""
    registry = cast("FeatureJobRegistry", model_registry)
    due = list(await maybe_await(registry.due_feature_jobs(as_of)))
    processed = 0
    features_stored = 0
    errors = 0

    for job in due:
        try:
            result: DataMaintenanceResult = await scheduler.run_once(
                strategy_run_id=strategy_run_id,
                as_of=as_of,
                feature_set_version=job.feature_set_version,
            )
            await maybe_await(registry.mark_job_completed(job.job_id, run_at=as_of, success=True))
            processed += 1
            features_stored += result.features_stored
            log.info(
                "maintenance_supervisor.job_ok",
                job_id=str(job.job_id),
                features_stored=result.features_stored,
                bars_ingested=result.bars_ingested,
            )
        except Exception as exc:
            errors += 1
            await maybe_await(registry.mark_job_completed(job.job_id, run_at=as_of, success=False))
            log.error(
                "maintenance_supervisor.job_failed",
                job_id=str(job.job_id),
                error=str(exc),
                exc_info=True,
            )

    return MaintenanceTick(
        as_of=as_of,
        jobs_processed=processed,
        features_stored=features_stored,
        errors=errors,
    )


__all__ = ["MaintenanceTick", "maybe_await", "run_due_feature_jobs"]
