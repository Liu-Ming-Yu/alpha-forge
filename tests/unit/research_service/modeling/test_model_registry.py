"""Unit tests for research model registry + feature job scheduling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_platform.services.research_service.modeling.registry.model_registry import (
    InMemoryModelRegistry,
)

_UTC = UTC


def test_model_registration_and_active_lookup() -> None:
    registry = InMemoryModelRegistry()
    now = datetime(2026, 1, 5, 14, 0, tzinfo=_UTC)
    model = registry.register_model(
        strategy_name="cross_sectional_equity_v1",
        model_version="0.1.0",
        feature_set_version="1.0.0",
        as_of=now,
    )
    active = registry.get_active_model("cross_sectional_equity_v1")
    assert active is not None
    assert active.model_id == model.model_id


def test_feature_job_due_and_reschedule() -> None:
    registry = InMemoryModelRegistry()
    now = datetime(2026, 1, 5, 14, 0, tzinfo=_UTC)
    model = registry.register_model(
        strategy_name="cross_sectional_equity_v1",
        model_version="0.1.0",
        feature_set_version="1.0.0",
        as_of=now,
    )
    job = registry.schedule_feature_job(
        model_id=model.model_id,
        strategy_name=model.strategy_name,
        feature_set_version=model.feature_set_version,
        interval_seconds=60.0,
        as_of=now,
    )
    assert [j.job_id for j in registry.due_feature_jobs(now)] == [job.job_id]

    registry.mark_job_completed(job.job_id, run_at=now, success=True)
    assert registry.due_feature_jobs(now + timedelta(seconds=30)) == []
    assert [j.job_id for j in registry.due_feature_jobs(now + timedelta(seconds=61))] == [
        job.job_id
    ]


def test_feature_job_failure_backoff() -> None:
    registry = InMemoryModelRegistry()
    now = datetime(2026, 1, 5, 14, 0, tzinfo=_UTC)
    model = registry.register_model(
        strategy_name="s",
        model_version="v",
        feature_set_version="f",
        as_of=now,
    )
    job = registry.schedule_feature_job(
        model_id=model.model_id,
        strategy_name=model.strategy_name,
        feature_set_version=model.feature_set_version,
        interval_seconds=120.0,
        as_of=now,
    )
    registry.mark_job_completed(job.job_id, run_at=now, success=False)
    due = registry.due_feature_jobs(now + timedelta(seconds=59))
    assert due == []
    due_later = registry.due_feature_jobs(now + timedelta(seconds=61))
    assert len(due_later) == 1
    assert due_later[0].job_id == job.job_id
    assert due_later[0].consecutive_failures == 1
