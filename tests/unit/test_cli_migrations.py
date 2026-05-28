"""Tests for CLI migration guardrails."""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from quant_platform.config import PlatformSettings, StorageSettings
from quant_platform.infrastructure.support import migrations

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_alembic_head_resolves_from_package_resources() -> None:
    assert migrations.packaged_head() == "028"


def test_packaged_alembic_chain_validates_offline() -> None:
    assert migrations.validate_packaged_migration_chain() == "028"


def test_packaged_alembic_config_uses_installed_psycopg3_driver() -> None:
    pytest.importorskip("alembic.config", reason="alembic not installed in this environment")
    dsn = "postgresql+psycopg://quant:quant@localhost:5432/quant_platform_test"
    expected = "postgresql+psycopg://quant:quant@localhost:5432/quant_platform_test"

    with migrations.alembic_config(dsn) as cfg:
        assert cfg.get_main_option("sqlalchemy.url") == expected


def test_repo_local_alembic_uses_packaged_migration_authority() -> None:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "alembic.ini", encoding="utf-8")
    assert parser.get("alembic", "script_location") == "src/quant_platform/alembic"


def test_root_alembic_tree_has_no_runtime_migration_files() -> None:
    assert not list((ROOT / "alembic").glob("*.py"))
    assert not list((ROOT / "alembic" / "versions").glob("*.py"))


def test_constraint_hardening_uses_execution_id_not_broker_order_uniqueness() -> None:
    migration = (
        ROOT / "src" / "quant_platform" / "alembic" / "versions" / "009_constraint_hardening.py"
    ).read_text(encoding="utf-8")

    assert "broker_execution_id" in migration
    assert "uq_fill_events_broker_execution" in migration
    assert "ix_fill_events_broker_order_id" not in migration
    assert "duplicate_object" in migration
    assert "ADD CONSTRAINT IF NOT EXISTS" not in migration


@pytest.mark.asyncio
async def test_postgres_runtime_guard_runs_when_dsn_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_verify(dsn: str) -> None:
        calls.append(dsn)

    monkeypatch.setattr(migrations, "verify_alembic_head", _fake_verify)
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(
            postgres_dsn="postgresql+psycopg://quant:quant@localhost/quant",
        ),
    )

    await migrations.verify_postgres_schema_if_configured(settings)

    assert calls == ["postgresql+psycopg://quant:quant@localhost/quant"]


@pytest.mark.asyncio
async def test_postgres_runtime_guard_skips_in_memory_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_verify(dsn: str) -> None:
        calls.append(dsn)

    monkeypatch.setattr(migrations, "verify_alembic_head", _fake_verify)
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn=""),
    )

    await migrations.verify_postgres_schema_if_configured(settings)

    assert calls == []
