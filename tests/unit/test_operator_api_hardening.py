"""Tests for the hardening fixes from code review.

Covers: config secret-scrub (keys + embedded credentials), store_false flag
reconstruction, job cancellation, the broker-sync graceful path, and static
SPA serving (redirect + build-instructions placeholder).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from quant_platform.config import ApiSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api import broker_sync
from quant_platform.views.operator_api.app import create_app
from quant_platform.views.operator_api.commands.catalog import reconstruct_argv
from quant_platform.views.operator_api.commands.jobs import JobStore
from quant_platform.views.operator_api.routers.console_config import _scrub

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        api=ApiSettings(operator_api_key="test-key"),
    )


# --- config scrub ---------------------------------------------------------


def test_scrub_drops_secret_keys() -> None:
    out = _scrub(
        {
            "port": 7497,
            "operator_api_key": "secret",
            "postgres_dsn": "x",
            "nested": {"redis_url": "y", "ok": 1},
        }
    )
    assert isinstance(out, dict)
    assert "operator_api_key" not in out
    assert "postgres_dsn" not in out
    assert out["port"] == 7497
    assert "redis_url" not in out["nested"]
    assert out["nested"]["ok"] == 1


def test_scrub_redacts_embedded_credentials() -> None:
    out = _scrub(
        {
            "endpoint": "postgresql://user:pass@host:5432/db",
            "plain": "http://host/path",
        }
    )
    assert isinstance(out, dict)
    assert out["endpoint"] == "***redacted***"
    assert out["plain"] == "http://host/path"  # no embedded creds → kept


# --- flag reconstruction (store_true and store_false) ---------------------


def test_reconstruct_store_true_flag() -> None:
    cmd = {
        "path": ["x"],
        "args": [
            {
                "dest": "verbose",
                "option_strings": ["--verbose"],
                "positional": False,
                "kind": "flag",
                "default": False,
            }
        ],
    }
    assert reconstruct_argv(cmd, {"verbose": False}) == ["x"]
    assert reconstruct_argv(cmd, {"verbose": True}) == ["x", "--verbose"]


def test_reconstruct_store_false_flag() -> None:
    cmd = {
        "path": ["x"],
        "args": [
            {
                "dest": "feature",
                "option_strings": ["--no-feature"],
                "positional": False,
                "kind": "flag",
                "default": True,
            }
        ],
    }
    assert reconstruct_argv(cmd, {"feature": True}) == ["x"]  # default → no flag
    assert reconstruct_argv(cmd, {"feature": False}) == ["x", "--no-feature"]  # flipped → flag


# --- job cancellation -----------------------------------------------------


def test_jobstore_cancel_terminates_process() -> None:
    store = JobStore()
    job = store.create(["x"], ["x"])

    class _FakeProc:
        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    proc = _FakeProc()
    store.mark_running(job, cast("subprocess.Popen[str]", proc))
    assert store.cancel(job.id) is True
    assert proc.terminated is True
    assert store.cancel("does-not-exist") is False


# --- numeric coercion (must handle Decimal money values) ------------------


def test_numeric_coercion_handles_decimal() -> None:
    from decimal import Decimal

    from quant_platform.bootstrap.broker.data_sync import _f
    from quant_platform.views.operator_api.backtest import _num

    assert _f(Decimal("1176976.56")) == 1176976.56
    assert _f("3.5") == 3.5
    assert _f(None) is None
    assert _f(object()) is None
    assert _num(Decimal("0.1")) == 0.1
    assert _num("bad") == 0.0


# --- broker sync graceful path -------------------------------------------


def test_sync_broker_graceful_without_ibapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_sync, "_ibapi_available", lambda: False)
    result = asyncio.run(broker_sync.sync_broker(_settings(), "paper"))
    assert result["connected"] is False
    assert "ibapi" in result["error"].lower()
    assert result["port"] == 7497  # still mode-resolved


# --- static SPA serving ---------------------------------------------------


def test_static_root_redirects_and_placeholder(tmp_path: Path) -> None:
    settings = _settings()
    settings.api.console_dist_dir = str(tmp_path / "absent_dist")
    client = TestClient(create_app(settings=settings))
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/app/"
    app_root = client.get("/app/", follow_redirects=False)
    assert app_root.status_code == 200
    assert "not built" in app_root.text.lower()
