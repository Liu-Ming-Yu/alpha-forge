"""Health, capabilities, readiness, and HTTP metrics routes."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import structlog
from fastapi.responses import JSONResponse, Response

from quant_platform.bootstrap.operator_api.queries import (
    latest_feature_vector_age_seconds,
    postgres_ready,
    render_operator_metrics,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext

log = structlog.get_logger(__name__)


async def health_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    if ctx.shutting_down[0]:
        return JSONResponse(content={"status": "draining"}, status_code=503)
    return JSONResponse(content={"status": "ok"})


async def operator_capabilities_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    return JSONResponse(content=ctx.capabilities_payload())


async def health_details_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    view = await ctx.builder.broker_health()
    data = asdict(view)
    last_hb = data.get("last_heartbeat_at")
    if last_hb is not None and hasattr(last_hb, "isoformat"):
        data["last_heartbeat_at"] = last_hb.isoformat()
    return JSONResponse(content=data)


async def health_ready_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    if ctx.shutting_down[0]:
        return JSONResponse(content={"status": "draining", "ready": False}, status_code=503)
    checks: dict[str, Any] = {"postgres": "skipped", "redis": "skipped"}
    ready = True
    if ctx.settings.storage.postgres_dsn:
        try:
            await postgres_ready(ctx.settings)
            checks["postgres"] = "ok"
        except Exception:
            checks["postgres"] = "error"
            ready = False
    if ctx.settings.storage.redis_url:
        try:
            from quant_platform.bootstrap.persistence.redis_factory import create_async_redis_client

            redis_client = create_async_redis_client(
                ctx.settings.storage.redis_url,
                decode_responses=True,
            )
            try:
                await redis_client.ping()
                checks["redis"] = "ok"
                stream_prefix = ctx.settings.storage.redis_stream_prefix
                max_pending = 0
                scanned_streams = 0
                try:
                    async for sk in redis_client.scan_iter(
                        match=f"{stream_prefix}:*",
                        count=100,
                    ):
                        scanned_streams += 1
                        if scanned_streams > 1000:
                            break
                        if sk.endswith(".dlq"):
                            continue
                        groups = await redis_client.xinfo_groups(sk)
                        for group in groups or []:
                            pending = int(group.get("pending", 0))
                            if pending > max_pending:
                                max_pending = pending
                except Exception as exc:
                    log.debug(
                        "operator_api.redis_pending_scan_failed",
                        error=str(exc),
                    )
                checks["event_bus_max_pending"] = max_pending
                checks["event_bus_streams_scanned"] = scanned_streams
            finally:
                await redis_client.aclose()
        except Exception:
            checks["redis"] = "error"
            ready = False
    if ctx.settings.storage.postgres_dsn:
        try:
            checks["feature_vector_age_seconds"] = await latest_feature_vector_age_seconds(
                ctx.settings
            )
        except Exception:
            checks["feature_vector_age_seconds"] = None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


def prometheus_metrics_response() -> Response:
    payload, content_type = render_operator_metrics()
    return Response(content=payload, media_type=content_type)


def register_health_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    protected_dependencies = ctx.protected_dependencies

    @app.get("/health")
    async def health() -> JSONResponse:
        return await health_response(ctx)

    @app.get("/operator/capabilities", dependencies=protected_dependencies)
    async def operator_capabilities() -> JSONResponse:
        return await operator_capabilities_response(ctx)

    @app.get("/health/details", dependencies=protected_dependencies)
    async def health_details() -> JSONResponse:
        return await health_details_response(ctx)

    @app.get("/health/ready", dependencies=protected_dependencies)
    async def health_ready() -> JSONResponse:
        return await health_ready_response(ctx)

    if getattr(ctx.settings.api, "expose_metrics", False):

        @app.get("/metrics", dependencies=protected_dependencies)
        async def prometheus_metrics() -> Response:
            return prometheus_metrics_response()
