"""Tests for the session-start model-registry preflight (Phase 3.4)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.config import PlatformSettings, RiskSettings, StorageSettings
from quant_platform.services.research_service.modeling.registry.model_registry import (
    RegisteredModel,
)
from quant_platform.session import model_registry_preflight


class _FakeRegistry:
    def __init__(self, active: RegisteredModel | None) -> None:
        self._active = active

    async def get_active_model(self, strategy_name: str) -> RegisteredModel | None:
        return self._active


class _StubSession:
    """Minimal session shape consumed by the preflight."""

    def __init__(self, settings: PlatformSettings) -> None:
        self.settings = settings


def _settings(*, dsn: str, strict: bool) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn=dsn),
        risk=RiskSettings(require_registered_model_match=strict),
    )


def _model(version: str) -> RegisteredModel:
    return RegisteredModel(
        model_id=uuid.uuid4(),
        strategy_name="xsec",
        model_version=version,
        feature_set_version="1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_preflight_no_postgres_is_noop() -> None:
    session = _StubSession(_settings(dsn="", strict=True))
    await model_registry_preflight(
        session, strategy_name="xsec", engine_version="1.2.3"
    )  # no raise, no registry needed


@pytest.mark.asyncio
async def test_preflight_mismatch_non_strict_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(_settings(dsn="postgresql+asyncpg://x", strict=False))
    fake = _FakeRegistry(_model("1.0.0"))
    monkeypatch.setattr(
        "quant_platform.infrastructure.postgres.model_registry.build_model_registry",
        lambda dsn: fake,
    )
    await model_registry_preflight(session, strategy_name="xsec", engine_version="2.0.0")


@pytest.mark.asyncio
async def test_preflight_mismatch_strict_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(_settings(dsn="postgresql+asyncpg://x", strict=True))
    fake = _FakeRegistry(_model("1.0.0"))
    monkeypatch.setattr(
        "quant_platform.infrastructure.postgres.model_registry.build_model_registry",
        lambda dsn: fake,
    )
    with pytest.raises(RuntimeError, match="preflight mismatch"):
        await model_registry_preflight(session, strategy_name="xsec", engine_version="2.0.0")


@pytest.mark.asyncio
async def test_preflight_match_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _StubSession(_settings(dsn="postgresql+asyncpg://x", strict=True))
    fake = _FakeRegistry(_model("2.0.0"))
    monkeypatch.setattr(
        "quant_platform.infrastructure.postgres.model_registry.build_model_registry",
        lambda dsn: fake,
    )
    await model_registry_preflight(session, strategy_name="xsec", engine_version="2.0.0")


@pytest.mark.asyncio
async def test_preflight_missing_active_strict_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(_settings(dsn="postgresql+asyncpg://x", strict=True))
    fake = _FakeRegistry(None)
    monkeypatch.setattr(
        "quant_platform.infrastructure.postgres.model_registry.build_model_registry",
        lambda dsn: fake,
    )
    with pytest.raises(RuntimeError, match="no active RegisteredModel"):
        await model_registry_preflight(session, strategy_name="xsec", engine_version="2.0.0")
