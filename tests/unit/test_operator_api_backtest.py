"""Unit tests for the backtest evidence reader + /v1/backtest endpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

from quant_platform.config import ApiSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api.app import create_app
from quant_platform.views.operator_api.backtest import (
    list_backtest_runs,
    load_backtest_result,
)

_H = {"X-API-Key": "test-key"}


def _fold(idx: int, start: str, end: str, ret: float, ic: float) -> dict:
    return {
        "fold_index": idx,
        "test_start": f"{start}T00:00:00+00:00",
        "test_end": f"{end}T00:00:00+00:00",
        "total_return": ret,
        "mean_ic": ic,
        "slippage_adjusted_sharpe": 1.0,
        "max_drawdown": -0.02,
        "turnover_avg": 0.01,
    }


def _write_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "research" / "backtest_demo"
    run_dir.mkdir(parents=True)
    evidence = {
        "arm": "demo_arm",
        "arm_category": "demo",
        "run_id": "abc123",
        "saved_at_utc": "2026-05-01T00:00:00+00:00",
        "metrics": {"ic_60d": 0.05, "max_drawdown": -0.03, "fold_negative_ic_streak": 2.0},
        "folds": [
            _fold(0, "2024-01-05", "2024-01-26", 0.10, 0.04),
            _fold(1, "2024-01-26", "2024-02-16", -0.05, -0.01),
            _fold(2, "2024-02-16", "2024-03-08", 0.10, 0.06),
        ],
    }
    (run_dir / "arm_demo.json").write_text(json.dumps(evidence), encoding="utf-8")


def _settings(tmp_path: Path) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(
            postgres_dsn="",
            redis_url="",
            event_bus_backend="in_memory",
            object_store_root=str(tmp_path),
        ),
        api=ApiSettings(operator_api_key="test-key"),
    )


def test_list_backtest_runs(tmp_path: Path) -> None:
    _write_evidence(tmp_path)
    runs = list_backtest_runs(_settings(tmp_path))
    assert len(runs) == 1
    run = runs[0]
    assert run["arm"] == "demo_arm"
    assert run["date_start"] == "2024-01-05"
    assert run["date_end"] == "2024-03-08"
    assert run["n_folds"] == 3
    # 1.10 * 0.95 * 1.10 - 1 == 0.1495
    assert run["total_return"] == pytest.approx(0.1495, abs=1e-6)


def test_load_backtest_result_builds_equity(tmp_path: Path) -> None:
    _write_evidence(tmp_path)
    settings = _settings(tmp_path)
    runs = list_backtest_runs(settings)
    result = load_backtest_result(settings, runs[0]["id"])
    assert result is not None
    points = result["points"]
    assert len(points) == 3
    assert points[0]["equity"] == pytest.approx(1.10, abs=1e-6)
    assert points[1]["equity"] == pytest.approx(1.045, abs=1e-6)
    assert points[2]["equity"] == pytest.approx(1.1495, abs=1e-6)
    # drawdown after the down fold is negative
    assert points[1]["drawdown"] < 0
    assert result["metrics"]["total_return"] == pytest.approx(0.1495, abs=1e-6)


def test_traversal_is_rejected(tmp_path: Path) -> None:
    _write_evidence(tmp_path)
    assert load_backtest_result(_settings(tmp_path), "../../../../etc/passwd") is None


def test_backtest_endpoints(tmp_path: Path) -> None:
    _write_evidence(tmp_path)
    client = TestClient(create_app(settings=_settings(tmp_path)))
    runs = client.get("/v1/backtest/runs", headers=_H)
    assert runs.status_code == 200
    run_id = runs.json()["runs"][0]["id"]
    result = client.get(f"/v1/backtest/result?run_id={run_id}", headers=_H)
    assert result.status_code == 200
    assert len(result.json()["points"]) == 3
    assert client.get("/v1/backtest/result?run_id=nope", headers=_H).status_code == 404
    assert client.get("/v1/backtest/runs").status_code == 401  # auth required
