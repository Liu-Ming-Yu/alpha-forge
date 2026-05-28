"""Unit tests for engine feature dataset freshness guard."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from quant_platform.config import PlatformSettings, V2Settings
from quant_platform.engines.feature_jobs.dataset_guard import load_required_feature_dataset_id

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


def _settings(*, require: bool, max_age_seconds: int = 3600) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        v2=V2Settings(
            enabled=True,
            require_feature_datasets=require,
            max_feature_age_seconds=max_age_seconds,
        ),
    )


def _session(*, dataset_catalog: object | None = object()) -> SimpleNamespace:
    instrument_id = uuid.uuid4()
    return SimpleNamespace(
        dataset_catalog=dataset_catalog,
        feature_repo=object(),
        contract_master=SimpleNamespace(
            list_active=lambda: [SimpleNamespace(instrument_id=instrument_id)]
        ),
        instrument_id=instrument_id,
    )


@pytest.mark.asyncio
async def test_feature_dataset_guard_is_noop_when_not_required() -> None:
    result = await load_required_feature_dataset_id(
        object(),
        settings=_settings(require=False),
        feature_set_version="daily-v1",
        as_of=_AS_OF,
    )

    assert result is None


@pytest.mark.asyncio
async def test_feature_dataset_guard_is_noop_without_catalog() -> None:
    result = await load_required_feature_dataset_id(
        _session(dataset_catalog=None),
        settings=_settings(require=True),
        feature_set_version="daily-v1",
        as_of=_AS_OF,
    )

    assert result is None


@pytest.mark.asyncio
async def test_feature_dataset_guard_returns_fresh_dataset_id() -> None:
    dataset_id = uuid.uuid4()
    session = _session()
    snapshot = SimpleNamespace(
        as_of=_AS_OF - timedelta(seconds=600),
        dataset=SimpleNamespace(dataset_id=dataset_id),
    )

    with patch(
        "quant_platform.services.research_service.feature_quality.snapshot.load_feature_snapshot",
        new=AsyncMock(return_value=snapshot),
    ) as loader:
        result = await load_required_feature_dataset_id(
            session,
            settings=_settings(require=True, max_age_seconds=3600),
            feature_set_version="daily-v1",
            as_of=_AS_OF,
        )

    assert result == dataset_id
    loader.assert_awaited_once()
    assert loader.await_args.kwargs["instrument_ids"] == [session.instrument_id]
    assert loader.await_args.kwargs["feature_set_version"] == "daily-v1"


@pytest.mark.asyncio
async def test_feature_dataset_guard_rejects_stale_snapshot() -> None:
    snapshot = SimpleNamespace(
        as_of=_AS_OF - timedelta(seconds=7200),
        dataset=SimpleNamespace(dataset_id=uuid.uuid4()),
    )

    with (
        patch(
            "quant_platform.services.research_service.feature_quality.snapshot.load_feature_snapshot",
            new=AsyncMock(return_value=snapshot),
        ),
        pytest.raises(RuntimeError, match="feature snapshot is stale"),
    ):
        await load_required_feature_dataset_id(
            _session(),
            settings=_settings(require=True, max_age_seconds=3600),
            feature_set_version="daily-v1",
            as_of=_AS_OF,
        )
