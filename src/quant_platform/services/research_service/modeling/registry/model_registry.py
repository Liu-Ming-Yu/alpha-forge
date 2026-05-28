"""In-memory model registry and feature-job scheduling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True)
class RegisteredModel:
    model_id: uuid.UUID
    strategy_name: str
    model_version: str
    feature_set_version: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)
    active: bool = True


@dataclass
class FeatureJob:
    job_id: uuid.UUID
    model_id: uuid.UUID
    strategy_name: str
    feature_set_version: str
    interval_seconds: float
    next_run_at: datetime
    last_run_at: datetime | None = None
    enabled: bool = True
    consecutive_failures: int = 0


class InMemoryModelRegistry:
    """Tracks active models and scheduled feature jobs per strategy."""

    def __init__(self) -> None:
        self._models: dict[uuid.UUID, RegisteredModel] = {}
        self._active_by_strategy: dict[str, uuid.UUID] = {}
        self._jobs: dict[uuid.UUID, FeatureJob] = {}

    def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModel:
        model = RegisteredModel(
            model_id=uuid.uuid4(),
            strategy_name=strategy_name,
            model_version=model_version,
            feature_set_version=feature_set_version,
            created_at=as_of,
            metadata=dict(metadata or {}),
            active=True,
        )
        self._models[model.model_id] = model
        self._active_by_strategy[strategy_name] = model.model_id
        return model

    def get_active_model(self, strategy_name: str) -> RegisteredModel | None:
        model_id = self._active_by_strategy.get(strategy_name)
        if model_id is None:
            return None
        return self._models.get(model_id)

    def schedule_feature_job(
        self,
        *,
        model_id: uuid.UUID,
        strategy_name: str,
        feature_set_version: str,
        interval_seconds: float,
        as_of: datetime,
    ) -> FeatureJob:
        job = FeatureJob(
            job_id=uuid.uuid4(),
            model_id=model_id,
            strategy_name=strategy_name,
            feature_set_version=feature_set_version,
            interval_seconds=interval_seconds,
            next_run_at=as_of,
        )
        self._jobs[job.job_id] = job
        return job

    def due_feature_jobs(self, as_of: datetime) -> list[FeatureJob]:
        return [job for job in self._jobs.values() if job.enabled and job.next_run_at <= as_of]

    def mark_job_completed(
        self,
        job_id: uuid.UUID,
        *,
        run_at: datetime,
        success: bool,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_run_at = run_at
        if success:
            job.consecutive_failures = 0
            job.next_run_at = run_at + timedelta(seconds=job.interval_seconds)
        else:
            job.consecutive_failures += 1
            retry_seconds = min(300.0, max(5.0, job.interval_seconds / 2.0))
            job.next_run_at = run_at + timedelta(seconds=retry_seconds)
