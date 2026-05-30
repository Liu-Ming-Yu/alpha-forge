"""Unit tests for the System / factor / alpha / model dashboard endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from quant_platform.config import ApiSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api.app import create_app

_H = {"X-API-Key": "test-key"}


def _client() -> TestClient:
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        api=ApiSettings(operator_api_key="test-key"),
    )
    return TestClient(create_app(settings=settings))


def test_system_status_reports_hardware() -> None:
    response = _client().get("/v1/system/status", headers=_H)
    assert response.status_code == 200
    data = response.json()
    assert data["platform"]
    assert data["python"]
    assert isinstance(data["gpus"], list)
    if data.get("psutil_available"):
        assert data["cpu"]["logical"] >= 1
        assert 0 <= data["memory"]["percent"] <= 100


def test_feature_families_introspection() -> None:
    data = _client().get("/v1/features/families", headers=_H).json()
    assert data["total_families"] >= 1
    assert data["total_features"] >= 1
    names = {f["name"] for f in data["families"]}
    assert names & {"price_volume", "fundamentals", "formulaic"}
    first = data["families"][0]
    assert {"name", "version", "feature_count", "features"} <= set(first)


def test_alpha_library_endpoint() -> None:
    data = _client().get("/v1/alpha/library", headers=_H).json()
    assert "classical" in data["source_weights"]
    assert isinstance(data["alphas"], list)
    assert "auto_promoted_count" in data


def test_model_registry_graceful_without_db() -> None:
    data = _client().get("/v1/models/registry", headers=_H).json()
    assert data["count"] == 0  # no Postgres configured in the test
    assert isinstance(data["models"], list)


def test_dashboard_endpoints_require_auth() -> None:
    client = _client()
    assert client.get("/v1/system/status").status_code == 401
    assert client.get("/v1/features/families").status_code == 401
    assert client.get("/v1/alpha/library").status_code == 401
    assert client.get("/v1/models/registry").status_code == 401
