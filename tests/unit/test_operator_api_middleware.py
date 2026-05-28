"""Operator API middleware ordering and auth-vs-rate-limit precedence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.views.operator_api.middleware import (
    install_operator_api_middlewares,
)


def _make_app(*, rate_limit: int, operator_api_key: str) -> FastAPI:
    app = FastAPI()

    @app.get("/protected")
    async def _protected() -> dict[str, bool]:
        return {"ok": True}

    settings = SimpleNamespace(api=SimpleNamespace(rate_limit_per_minute=rate_limit))
    clock = SimpleNamespace(now=lambda: None)
    install_operator_api_middlewares(
        app,
        settings=settings,
        clock=clock,
        v2_auth_repo=None,
        operator_api_key=operator_api_key,
    )
    return app


def test_unauthenticated_request_returns_401_before_rate_limit() -> None:
    """Auth must wrap the rate limiter.

    Regression: Starlette's ``add_middleware`` inserts at index 0, so the
    *last* middleware registered runs *first*. The prior registration
    order put rate limiting outside auth, so an attacker without an API
    key could exhaust the rate-limit bucket before ever being rejected
    as unauthenticated.
    """
    app = _make_app(rate_limit=2, operator_api_key="secret")
    client = TestClient(app)

    # Burn more requests than the bucket allows; every one must be 401.
    for _ in range(5):
        response = client.get("/protected")
        assert response.status_code == 401, (
            f"unauthenticated request received {response.status_code}; "
            "rate limiter must not run before auth"
        )


def test_authenticated_request_is_allowed() -> None:
    app = _make_app(rate_limit=120, operator_api_key="secret")
    client = TestClient(app)

    response = client.get("/protected", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
