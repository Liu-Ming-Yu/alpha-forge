"""Unit tests for operator API authentication behavior."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from fastapi import FastAPI

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from quant_platform.config import ApiSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api.app import create_app


def _settings_with_api_key(
    api_key: str,
    *,
    allow_unauthenticated: bool = False,
    acknowledge_unauthenticated_risk: bool = False,
    expose_metrics: bool = False,
) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        api=ApiSettings(
            operator_api_key=api_key,
            allow_unauthenticated=allow_unauthenticated,
            acknowledge_unauthenticated_risk=acknowledge_unauthenticated_risk,
            expose_metrics=expose_metrics,
        ),
    )


def _get(
    app: FastAPI,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async def _request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path, headers=headers)

    return asyncio.run(_request())


def test_health_route_is_public_when_api_key_configured() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_details_route_requires_auth_when_api_key_configured() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/health/details")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_health_ready_route_requires_auth_when_api_key_configured() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/health/ready")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_health_ready_reports_skipped_backends_without_sensitive_details() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/health/ready", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": "skipped", "redis": "skipped"},
    }


def test_health_ready_uses_scan_and_configured_redis_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _FakeRedisClient:
        async def ping(self) -> bool:
            return True

        async def scan_iter(self, *, match: str, count: int):
            seen["match"] = match
            seen["count"] = count
            yield "custom:events:OrderSubmitted"

        async def xinfo_groups(self, stream: str) -> list[dict[str, int]]:
            seen["stream"] = stream
            return [{"pending": 3}]

        async def aclose(self) -> None:
            seen["closed"] = True

    redis_asyncio = types.ModuleType("redis.asyncio")
    redis_asyncio.from_url = lambda *args, **kwargs: _FakeRedisClient()  # type: ignore[attr-defined]
    redis_module = types.ModuleType("redis")
    redis_module.asyncio = redis_asyncio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setitem(sys.modules, "redis.asyncio", redis_asyncio)

    app = create_app(
        settings=PlatformSettings(
            _env_file=None,
            storage=StorageSettings(
                postgres_dsn="",
                redis_url="redis://fake",
                event_bus_backend="redis_streams",
                redis_stream_prefix="custom:events",
            ),
            api=ApiSettings(operator_api_key="secret-key"),
        )
    )

    response = _get(app, "/health/ready", headers={"X-API-Key": "secret-key"})

    assert response.status_code == 200
    checks = response.json()["checks"]
    assert checks["redis"] == "ok"
    assert checks["event_bus_max_pending"] == 3
    assert checks["event_bus_streams_scanned"] == 1
    assert seen == {
        "match": "custom:events:*",
        "count": 100,
        "stream": "custom:events:OrderSubmitted",
        "closed": True,
    }


def test_openapi_schema_is_disabled_by_default() -> None:
    # With global auth middleware, /openapi.json returns 401 without a key.
    # Verify it's truly absent by checking that even a valid key yields 404.
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/openapi.json", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 404


def test_cash_route_requires_auth_when_api_key_configured() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/cash")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_cash_route_accepts_x_api_key_header() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/cash", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200


def test_cash_route_accepts_bearer_token() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key"))
    response = _get(app, "/cash", headers={"Authorization": "Bearer secret-key"})
    assert response.status_code == 200


def test_app_refuses_to_start_without_api_key_by_default() -> None:
    with pytest.raises(RuntimeError, match="refuses to start without"):
        create_app(settings=_settings_with_api_key(""))


def test_cash_route_is_public_with_explicit_unauthenticated_escape_hatch() -> None:
    app = create_app(
        settings=_settings_with_api_key(
            "",
            allow_unauthenticated=True,
            acknowledge_unauthenticated_risk=True,
        )
    )
    response = _get(app, "/cash")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# R-OBS-01: /metrics must be gated on the operator API key when expose_metrics
# is on, so a mis-deployed reverse-proxy allow-list cannot leak strategy
# performance telemetry.
# ---------------------------------------------------------------------------


def test_metrics_route_requires_auth_when_api_key_configured() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key", expose_metrics=True))
    response = _get(app, "/metrics")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_metrics_route_accepts_valid_api_key() -> None:
    app = create_app(settings=_settings_with_api_key("secret-key", expose_metrics=True))
    response = _get(app, "/metrics", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200


def test_metrics_route_is_public_only_with_allow_unauthenticated_escape_hatch() -> None:
    """With no API key *and* the explicit escape hatch, /metrics remains open.
    This matches the dev / smoke-test workflow where the operator has
    already acknowledged the risk via the dual unauthenticated opt-in."""
    app = create_app(
        settings=_settings_with_api_key(
            "",
            allow_unauthenticated=True,
            acknowledge_unauthenticated_risk=True,
            expose_metrics=True,
        )
    )
    response = _get(app, "/metrics")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# R-OBS-02: dual-opt-in for unauthenticated startup.  A single flag must
# not be enough to expose the protected surface.
# ---------------------------------------------------------------------------


def test_startup_refuses_with_allow_unauthenticated_alone() -> None:
    """allow_unauthenticated=true without the acknowledgement flag must refuse."""
    with pytest.raises(RuntimeError, match="ACKNOWLEDGE_UNAUTHENTICATED_RISK"):
        create_app(settings=_settings_with_api_key("", allow_unauthenticated=True))


def test_startup_refuses_with_acknowledgement_alone() -> None:
    """Acknowledgement alone (without allow_unauthenticated) still refuses."""
    with pytest.raises(RuntimeError, match="refuses to start without"):
        create_app(settings=_settings_with_api_key("", acknowledge_unauthenticated_risk=True))


def test_startup_succeeds_with_both_flags(capsys: CaptureFixture[str]) -> None:
    """Both flags set -> startup proceeds with the loud SECURITY_WARNING log.

    The positive path is already exercised by
    ``test_cash_route_is_public_with_explicit_unauthenticated_escape_hatch``
    above; this test pins the boot-path so a future refactor cannot
    silently regress it back to a one-flag gate.
    """
    app = create_app(
        settings=_settings_with_api_key(
            "",
            allow_unauthenticated=True,
            acknowledge_unauthenticated_risk=True,
        )
    )
    response = _get(app, "/health")
    assert response.status_code == 200

    captured = capsys.readouterr()
    assert "operator_api.SECURITY_WARNING.unauthenticated_enabled" in captured.out
    assert "QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true" in captured.out
