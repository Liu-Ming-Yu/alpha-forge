"""Scheduled feature-job execution for engine runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import structlog

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.runtime.registry import maybe_await

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable
    from datetime import datetime

    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
        DataMaintenanceResult,
    )
    from quant_platform.services.research_service.modeling.registry.model_registry import FeatureJob


class FeatureJobRegistry(Protocol):
    def due_feature_jobs(
        self,
        as_of: datetime,
    ) -> list[FeatureJob] | Awaitable[list[FeatureJob]]: ...

    def mark_job_completed(
        self,
        job_id: uuid.UUID,
        *,
        run_at: datetime,
        success: bool,
    ) -> None | Awaitable[None]: ...


class FeatureMaintenanceScheduler(Protocol):
    def run_once(
        self,
        *,
        strategy_run_id: uuid.UUID,
        as_of: datetime,
        feature_set_version: str,
    ) -> Awaitable[DataMaintenanceResult]: ...


log = structlog.get_logger(__name__)


async def run_due_feature_jobs(
    *,
    model_registry: FeatureJobRegistry,
    maintenance_scheduler: FeatureMaintenanceScheduler | None,
    strategy_run: StrategyRun | None,
    as_of: datetime,
    engine_name: str,
    halt_on_stale_features: bool,
    fail_on_error: bool = False,
) -> dict[uuid.UUID, dict[str, float]]:
    """Run due feature jobs and return the latest generated feature rows."""
    if maintenance_scheduler is None or strategy_run is None:
        return {}
    due_jobs = cast(
        "list[FeatureJob]",
        await maybe_await(model_registry.due_feature_jobs(as_of)),
    )
    if not due_jobs:
        return {}

    latest_features: dict[uuid.UUID, dict[str, float]] = {}
    for feature_set_version, jobs in _jobs_by_feature_set(due_jobs).items():
        job = jobs[0]
        if len(jobs) > 1:
            log.warning(
                "engine_runner.duplicate_due_feature_jobs",
                engine=engine_name,
                feature_set_version=feature_set_version,
                job_ids=[str(due_job.job_id) for due_job in jobs],
                detail="coalescing duplicate due jobs into one feature maintenance run",
            )
        try:
            result = await maintenance_scheduler.run_once(
                strategy_run_id=strategy_run.run_id,
                as_of=as_of,
                feature_set_version=feature_set_version,
            )
            latest_features = result.feature_data or latest_features
            if result.features_stored == 0:
                log.error(
                    "engine_runner.feature_job_no_output",
                    engine=engine_name,
                    job_id=str(job.job_id),
                    detail="feature job ran but stored zero features - signals may be stale",
                )
                if halt_on_stale_features or fail_on_error:
                    raise DataStalenessError(
                        f"feature job {job.job_id} produced no features",
                    )
            await _mark_jobs_completed(model_registry, jobs, run_at=as_of, success=True)
            if result.features_stored > 0:
                log.info(
                    "engine_runner.feature_job_completed",
                    engine=engine_name,
                    job_id=str(job.job_id),
                    bars_ingested=result.bars_ingested,
                    liquidity_profiles=result.liquidity_profiles_updated,
                    features_stored=result.features_stored,
                )
        except Exception as exc:
            await _mark_jobs_completed(model_registry, jobs, run_at=as_of, success=False)
            log.error(
                "engine_runner.feature_job_failed",
                engine=engine_name,
                job_id=str(job.job_id),
                error=str(exc),
            )
            if halt_on_stale_features or fail_on_error:
                raise
    return latest_features


def _jobs_by_feature_set(
    due_jobs: list[FeatureJob],
) -> dict[str, list[FeatureJob]]:
    grouped: dict[str, list[FeatureJob]] = {}
    for job in due_jobs:
        grouped.setdefault(job.feature_set_version, []).append(job)
    return grouped


async def _mark_jobs_completed(
    model_registry: FeatureJobRegistry,
    jobs: list[FeatureJob],
    *,
    run_at: datetime,
    success: bool,
) -> None:
    for job in jobs:
        await maybe_await(
            model_registry.mark_job_completed(
                job.job_id,
                run_at=run_at,
                success=success,
            )
        )
