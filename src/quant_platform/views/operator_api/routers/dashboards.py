"""System / model / factor / alpha dashboard-data routes (read-only).

* ``GET /v1/system/status``     — host hardware + GPU + process metrics.
* ``GET /v1/features/families`` — feature factory families + feature specs.
* ``GET /v1/alpha/library``     — alpha blend config + formulaic alpha library.
* ``GET /v1/models/registry``   — registered/promoted models (Postgres-backed).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from quant_platform.views.operator_api.insights import (
    alpha_library,
    feature_families,
    model_registry,
)
from quant_platform.views.operator_api.system_status import system_status

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


def register_dashboard_data_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    deps = ctx.protected_dependencies

    @app.get("/v1/system/status", dependencies=deps)
    async def system_status_route() -> JSONResponse:
        # nvidia-smi + psutil are blocking; run off the event loop.
        return JSONResponse(content=await asyncio.to_thread(system_status, ctx.settings))

    @app.get("/v1/features/families", dependencies=deps)
    async def feature_families_route() -> JSONResponse:
        return JSONResponse(content=await asyncio.to_thread(feature_families))

    @app.get("/v1/alpha/library", dependencies=deps)
    async def alpha_library_route() -> JSONResponse:
        return JSONResponse(content=await asyncio.to_thread(alpha_library, ctx.settings))

    @app.get("/v1/models/registry", dependencies=deps)
    async def model_registry_route() -> JSONResponse:
        return JSONResponse(content=await model_registry(ctx.settings))
