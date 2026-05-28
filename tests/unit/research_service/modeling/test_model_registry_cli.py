"""Tests for ``model_registry_cli`` dispatch.

Uses a fake registry that mirrors the small subset of
``PostgresModelRegistry`` the CLI depends on so the tests do not
require Postgres.
"""

from __future__ import annotations

import argparse
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from quant_platform.config import PlatformSettings, StorageSettings
from quant_platform.services.research_service.modeling.registry.cli import dispatch

if TYPE_CHECKING:
    from pathlib import Path


class _FakeRegistry:
    def __init__(self) -> None:
        self.models: dict[tuple[str, str], SimpleNamespace] = {}
        self.active: dict[str, str] = {}

    async def list_models(self, strategy_name: str | None = None):
        return [v for (sn, _), v in self.models.items() if not strategy_name or sn == strategy_name]

    async def get_model(self, strategy_name, model_version):
        return self.models.get((strategy_name, model_version))

    async def get_active_model(self, strategy_name):
        v = self.active.get(strategy_name)
        return self.models.get((strategy_name, v)) if v else None

    async def register_model(
        self, *, strategy_name, model_version, feature_set_version, as_of, metadata
    ):
        for (sn, mv), m in list(self.models.items()):
            if sn == strategy_name and m.active:
                m.active = False
        m = SimpleNamespace(
            model_id=uuid.uuid4(),
            strategy_name=strategy_name,
            model_version=model_version,
            feature_set_version=feature_set_version,
            created_at=as_of,
            metadata=metadata,
            active=True,
        )
        self.models[(strategy_name, model_version)] = m
        self.active[strategy_name] = model_version
        return m

    async def retire_model(self, strategy_name):
        v = self.active.pop(strategy_name, None)
        if v is None:
            return 0
        self.models[(strategy_name, v)].active = False
        return 1

    async def rollback_to_version(self, strategy_name, target_version):
        key = (strategy_name, target_version)
        if key not in self.models:
            raise LookupError(f"not found: {strategy_name}@{target_version}")
        for (sn, mv), m in self.models.items():
            m.active = sn == strategy_name and mv == target_version
        self.active[strategy_name] = target_version
        return self.models[key]


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://fake/fake"),
    )


@pytest.fixture
def patched_registry(monkeypatch):
    fake = _FakeRegistry()

    def _build(_dsn):
        return fake

    import quant_platform.services.research_service.modeling.registry.cli as mod

    monkeypatch.setattr(mod, "build_model_registry", _build)
    return fake


@pytest.mark.asyncio
async def test_promote_inserts_active_row(patched_registry, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"foo": 1}')
    args = argparse.Namespace(
        name="strategy_a",
        version="v1",
        engine_version="e1",
        feature_set_version="features-v1",
        config_path=cfg,
        metadata_path=None,
        artifact_manifest=None,
    )
    await dispatch(settings=_settings(), subcommand="promote", args=args)
    active = await patched_registry.get_active_model("strategy_a")
    assert active is not None
    assert active.model_version == "v1"
    assert active.feature_set_version == "features-v1"


@pytest.mark.asyncio
async def test_rollback_restores_prior_version(patched_registry, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    for version in ("v1", "v2"):
        await dispatch(
            settings=_settings(),
            subcommand="promote",
            args=argparse.Namespace(
                name="strategy_a",
                version=version,
                engine_version="e1",
                feature_set_version="features-v1",
                config_path=cfg,
                metadata_path=None,
                artifact_manifest=None,
            ),
        )
    assert (await patched_registry.get_active_model("strategy_a")).model_version == "v2"
    await dispatch(
        settings=_settings(),
        subcommand="rollback",
        args=argparse.Namespace(name="strategy_a", to_version="v1"),
    )
    assert (await patched_registry.get_active_model("strategy_a")).model_version == "v1"


@pytest.mark.asyncio
async def test_rollback_unknown_version_fails(patched_registry) -> None:
    with pytest.raises(SystemExit):
        await dispatch(
            settings=_settings(),
            subcommand="rollback",
            args=argparse.Namespace(name="strategy_a", to_version="missing"),
        )


@pytest.mark.asyncio
async def test_promote_stores_boosting_manifest_metadata(
    patched_registry,
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"validation_ic": 0.12}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "model_type": "xgboost_ranker",
          "model_version": "xgb-v1",
          "feature_set_version": "features-v1",
          "feature_schema_hash": "abc",
          "xgboost_version": "3.2.0",
          "device": "cpu",
          "objective": "rank:pairwise",
          "metrics_path": "metrics.json"
        }
        """,
        encoding="utf-8",
    )

    await dispatch(
        settings=_settings(),
        subcommand="promote",
        args=argparse.Namespace(
            name="strategy_a",
            version="v1",
            engine_version="e1",
            feature_set_version="features-v1",
            config_path=None,
            metadata_path=None,
            artifact_manifest=manifest,
        ),
    )

    active = await patched_registry.get_active_model("strategy_a")
    assert active.metadata["artifact_manifest"] == str(manifest)
    assert active.metadata["boosting"]["feature_schema_hash"] == "abc"
    assert active.metadata["boosting"]["metrics"]["validation_ic"] == 0.12


@pytest.mark.asyncio
async def test_retire_with_no_active_errors(patched_registry) -> None:
    with pytest.raises(SystemExit):
        await dispatch(
            settings=_settings(),
            subcommand="retire",
            args=argparse.Namespace(name="strategy_a"),
        )


@pytest.mark.asyncio
async def test_dispatch_without_dsn_errors() -> None:
    with pytest.raises(SystemExit, match="POSTGRES_DSN"):
        await dispatch(
            settings=PlatformSettings(
                _env_file=None,
                storage=StorageSettings(postgres_dsn=""),
            ),
            subcommand="list",
            args=argparse.Namespace(),
        )
