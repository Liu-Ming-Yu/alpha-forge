"""Unit tests for scheduled engine feature-job execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.feature_jobs.job_runtime import run_due_feature_jobs

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_due_feature_jobs_noops_without_scheduler() -> None:
    registry = SimpleNamespace(due_feature_jobs=AsyncMock())

    result = await run_due_feature_jobs(
        model_registry=registry,
        maintenance_scheduler=None,
        strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
        as_of=_AS_OF,
        engine_name="equity",
        halt_on_stale_features=False,
    )

    assert result == {}
    registry.due_feature_jobs.assert_not_called()


@pytest.mark.asyncio
async def test_due_feature_jobs_runs_due_job_and_marks_success() -> None:
    instrument_id = uuid.uuid4()
    job = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    registry = SimpleNamespace(
        due_feature_jobs=AsyncMock(return_value=[job]),
        mark_job_completed=AsyncMock(),
    )
    scheduler = SimpleNamespace(
        run_once=AsyncMock(
            return_value=SimpleNamespace(
                feature_data={instrument_id: {"momentum": 1.0}},
                features_stored=1,
                bars_ingested=2,
                liquidity_profiles_updated=3,
            )
        )
    )
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())

    result = await run_due_feature_jobs(
        model_registry=registry,
        maintenance_scheduler=scheduler,
        strategy_run=strategy_run,
        as_of=_AS_OF,
        engine_name="equity",
        halt_on_stale_features=False,
    )

    assert result == {instrument_id: {"momentum": 1.0}}
    scheduler.run_once.assert_awaited_once_with(
        strategy_run_id=strategy_run.run_id,
        as_of=_AS_OF,
        feature_set_version="daily-v1",
    )
    registry.mark_job_completed.assert_awaited_once_with(
        job.job_id,
        run_at=_AS_OF,
        success=True,
    )


@pytest.mark.asyncio
async def test_due_feature_jobs_coalesces_duplicate_feature_set_jobs() -> None:
    instrument_id = uuid.uuid4()
    first = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    second = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    registry = SimpleNamespace(
        due_feature_jobs=AsyncMock(return_value=[first, second]),
        mark_job_completed=AsyncMock(),
    )
    scheduler = SimpleNamespace(
        run_once=AsyncMock(
            return_value=SimpleNamespace(
                feature_data={instrument_id: {"momentum": 1.0}},
                features_stored=1,
                bars_ingested=2,
                liquidity_profiles_updated=3,
            )
        )
    )
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())

    result = await run_due_feature_jobs(
        model_registry=registry,
        maintenance_scheduler=scheduler,
        strategy_run=strategy_run,
        as_of=_AS_OF,
        engine_name="equity",
        halt_on_stale_features=False,
    )

    assert result == {instrument_id: {"momentum": 1.0}}
    scheduler.run_once.assert_awaited_once_with(
        strategy_run_id=strategy_run.run_id,
        as_of=_AS_OF,
        feature_set_version="daily-v1",
    )
    assert registry.mark_job_completed.await_count == 2
    registry.mark_job_completed.assert_any_await(
        first.job_id,
        run_at=_AS_OF,
        success=True,
    )
    registry.mark_job_completed.assert_any_await(
        second.job_id,
        run_at=_AS_OF,
        success=True,
    )


@pytest.mark.asyncio
async def test_due_feature_jobs_marks_failed_job_without_aborting_cycle() -> None:
    job = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    registry = SimpleNamespace(
        due_feature_jobs=AsyncMock(return_value=[job]),
        mark_job_completed=AsyncMock(),
    )
    scheduler = SimpleNamespace(run_once=AsyncMock(side_effect=RuntimeError("offline")))

    result = await run_due_feature_jobs(
        model_registry=registry,
        maintenance_scheduler=scheduler,
        strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
        as_of=_AS_OF,
        engine_name="equity",
        halt_on_stale_features=False,
    )

    assert result == {}
    registry.mark_job_completed.assert_awaited_once_with(
        job.job_id,
        run_at=_AS_OF,
        success=False,
    )


@pytest.mark.asyncio
async def test_due_feature_jobs_reraises_failures_when_fail_closed() -> None:
    job = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    registry = SimpleNamespace(
        due_feature_jobs=AsyncMock(return_value=[job]),
        mark_job_completed=AsyncMock(),
    )
    scheduler = SimpleNamespace(run_once=AsyncMock(side_effect=RuntimeError("offline")))

    with pytest.raises(RuntimeError, match="offline"):
        await run_due_feature_jobs(
            model_registry=registry,
            maintenance_scheduler=scheduler,
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            as_of=_AS_OF,
            engine_name="equity",
            halt_on_stale_features=False,
            fail_on_error=True,
        )

    registry.mark_job_completed.assert_awaited_once_with(
        job.job_id,
        run_at=_AS_OF,
        success=False,
    )


@pytest.mark.asyncio
async def test_due_feature_jobs_reraises_zero_output_when_fail_closed() -> None:
    job = SimpleNamespace(job_id=uuid.uuid4(), feature_set_version="daily-v1")
    registry = SimpleNamespace(
        due_feature_jobs=AsyncMock(return_value=[job]),
        mark_job_completed=AsyncMock(),
    )
    scheduler = SimpleNamespace(
        run_once=AsyncMock(
            return_value=SimpleNamespace(
                feature_data={},
                features_stored=0,
                bars_ingested=0,
                liquidity_profiles_updated=0,
            )
        )
    )

    with pytest.raises(DataStalenessError, match="produced no features"):
        await run_due_feature_jobs(
            model_registry=registry,
            maintenance_scheduler=scheduler,
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            as_of=_AS_OF,
            engine_name="equity",
            halt_on_stale_features=False,
            fail_on_error=True,
        )

    registry.mark_job_completed.assert_awaited_once_with(
        job.job_id,
        run_at=_AS_OF,
        success=False,
    )
