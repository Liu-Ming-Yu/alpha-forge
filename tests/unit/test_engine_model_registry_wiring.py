"""Unit tests for engine model-registry wiring."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.framework.model_registry_wiring import (
    register_engine_model_and_schedule_job,
)
from quant_platform.engines.framework.types import RunMode

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


class _SyncRegistry:
    def __init__(self, *, created_at: datetime = _AS_OF) -> None:
        self.model = SimpleNamespace(
            model_id=uuid.uuid4(),
            feature_set_version="daily-v1",
            strategy_name="equity",
            model_version="1.2.3",
            created_at=created_at,
        )
        self.job = SimpleNamespace(job_id=uuid.uuid4())
        self.register_kwargs: dict[str, object] | None = None
        self.schedule_kwargs: dict[str, object] | None = None

    def register_model(self, **kwargs: object) -> SimpleNamespace:
        self.register_kwargs = kwargs
        return self.model

    def schedule_feature_job(self, **kwargs: object) -> SimpleNamespace:
        self.schedule_kwargs = kwargs
        return self.job


@pytest.mark.asyncio
async def test_register_engine_model_and_schedule_job_records_engine_metadata() -> None:
    registry = _SyncRegistry()

    job = await register_engine_model_and_schedule_job(
        registry,
        engine_name="equity",
        engine_version="1.2.3",
        feature_set_version="daily-v1",
        run_mode=RunMode.PAPER,
        max_positions=15,
        interval_seconds=3600,
        as_of=_AS_OF,
        max_model_age_hours=24,
    )

    assert job is registry.job
    assert registry.register_kwargs is not None
    assert registry.register_kwargs["metadata"] == {
        "run_mode": "paper",
        "max_positions": 15,
    }
    assert registry.schedule_kwargs is not None
    assert registry.schedule_kwargs["model_id"] == registry.model.model_id
    assert registry.schedule_kwargs["interval_seconds"] == 3600


@pytest.mark.asyncio
async def test_register_engine_model_and_schedule_job_rejects_stale_model() -> None:
    registry = _SyncRegistry(created_at=_AS_OF - timedelta(hours=25))

    with pytest.raises(DataStalenessError, match="registered model is stale"):
        await register_engine_model_and_schedule_job(
            registry,
            engine_name="equity",
            engine_version="1.2.3",
            feature_set_version="daily-v1",
            run_mode=RunMode.PAPER,
            max_positions=15,
            interval_seconds=3600,
            as_of=_AS_OF,
            max_model_age_hours=24,
        )
