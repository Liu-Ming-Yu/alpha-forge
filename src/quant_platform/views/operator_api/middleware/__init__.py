"""HTTP middleware registration for the operator API."""

from __future__ import annotations

import secrets
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

import structlog
from fastapi import Request
from starlette.responses import Response

from quant_platform.bootstrap.operator_api.queries import (
    authorize_operator_viewer_api_key,
    record_operator_http_request,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import Clock, OperatorAuthRepository

_AUTH_PUBLIC_PATHS = frozenset({"/health", "/console/info"})


def _is_public_path(path: str) -> bool:
    """Paths served without authentication: liveness, console bootstrap, and
    the static SPA shell (``/`` and ``/app**``).  The shell must load before
    the operator supplies a key; every protected JSON endpoint stays gated."""
    return path in _AUTH_PUBLIC_PATHS or path == "/" or path == "/app" or path.startswith("/app/")


MiddlewareCallNext = Callable[[Request], Awaitable[Response]]
MiddlewareCallable = Callable[[Request, MiddlewareCallNext], Awaitable[Response]]
MiddlewareDecorator = Callable[[MiddlewareCallable], MiddlewareCallable]


class MiddlewareApp(Protocol):
    def middleware(self, middleware_type: str) -> MiddlewareDecorator: ...


def install_operator_api_middlewares(
    app: MiddlewareApp,
    *,
    settings: PlatformSettings,
    clock: Clock,
    v2_auth_repo: OperatorAuthRepository | None,
    operator_api_key: str,
) -> None:
    """Install auth, throttling, metrics, and correlation middleware.

    Starlette's ``add_middleware`` (which ``@app.middleware`` calls) inserts
    each middleware at index 0 of the user middleware list, so the
    *last* middleware registered runs *first* on incoming requests.
    Registration order below is therefore deliberately reversed from
    desired execution order:

        correlation_id  →  metrics  →  rate_limit  →  auth  →  router

    Auth is the outermost wrapper so unauthenticated traffic is rejected
    before consuming the rate-limit bucket or any downstream resource.
    """
    _install_correlation_id_middleware(app)
    _install_http_metrics_middleware(app)
    _install_rate_limit_middleware(app, rate_limit=int(settings.api.rate_limit_per_minute))
    _install_auth_middleware(
        app,
        clock=clock,
        v2_auth_repo=v2_auth_repo,
        operator_api_key=operator_api_key,
    )


def _install_auth_middleware(
    app: MiddlewareApp,
    *,
    clock: Clock,
    v2_auth_repo: OperatorAuthRepository | None,
    operator_api_key: str,
) -> None:
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next: MiddlewareCallNext) -> Response:
        if _is_public_path(request.url.path):
            return await call_next(request)
        if not operator_api_key:
            return await call_next(request)

        raw_key = _extract_api_key(request)
        if raw_key and secrets.compare_digest(raw_key, operator_api_key):
            return await call_next(request)

        if raw_key and v2_auth_repo is not None:
            try:
                record = await authorize_operator_viewer_api_key(
                    raw_key=raw_key,
                    repository=v2_auth_repo,
                    as_of=clock.now(),
                )
            except Exception:
                record = None
            if record is not None:
                return await call_next(request)

        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


def _extract_api_key(request: Request) -> str:
    raw_key = request.headers.get("X-API-Key", "")
    authorization = request.headers.get("Authorization", "")
    if not raw_key and authorization.lower().startswith("bearer "):
        return str(authorization[7:].strip())
    return str(raw_key)


def _install_rate_limit_middleware(app: MiddlewareApp, *, rate_limit: int) -> None:
    if rate_limit <= 0:
        return

    from fastapi.responses import JSONResponse

    bucket_capacity = float(rate_limit)
    bucket_refill_per_sec = bucket_capacity / 60.0
    buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
    max_buckets = 10_000

    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next: MiddlewareCallNext) -> Response:
        # Public/static console paths and Prometheus scrape do not consume the
        # per-key API budget (SPA asset loads + live polling stay independent).
        if request.url.path == "/metrics" or _is_public_path(request.url.path):
            return await call_next(request)

        key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "")
            or (request.client.host if request.client else "unknown")
        )
        if key in buckets:
            tokens, last = buckets.pop(key)
        else:
            tokens, last = bucket_capacity, time.monotonic()
        now = time.monotonic()
        tokens = min(bucket_capacity, tokens + (now - last) * bucket_refill_per_sec)
        if tokens < 1.0:
            buckets[key] = (tokens, now)
            _trim_buckets(buckets, max_buckets)
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": "1"},
            )
        buckets[key] = (tokens - 1.0, now)
        _trim_buckets(buckets, max_buckets)
        return await call_next(request)


def _trim_buckets(
    buckets: OrderedDict[str, tuple[float, float]],
    max_buckets: int,
) -> None:
    if len(buckets) > max_buckets:
        buckets.popitem(last=False)


def _install_http_metrics_middleware(app: MiddlewareApp) -> None:

    @app.middleware("http")
    async def _http_metrics_middleware(request: Request, call_next: MiddlewareCallNext) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        record_operator_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration_seconds=time.monotonic() - start,
        )
        return response


def _install_correlation_id_middleware(app: MiddlewareApp) -> None:

    @app.middleware("http")
    async def _correlation_id_middleware(
        request: Request,
        call_next: MiddlewareCallNext,
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(correlation_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("correlation_id")
        response.headers["X-Request-ID"] = request_id
        return response
