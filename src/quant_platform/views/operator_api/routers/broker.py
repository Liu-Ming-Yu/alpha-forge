"""TWS/IB-Gateway connection + data-sync routes (mode-aware, read-only).

* ``GET  /v1/broker/connection`` — the resolved connection target for the mode
  (paper → paper port, live → live port). No socket is opened.
* ``POST /v1/broker/sync``       — connect at that target and pull health,
  account (NAV/cash), positions, and open orders. Degrades gracefully when
  ``ibapi`` is absent or TWS is not running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from quant_platform.views.operator_api.broker_sync import connection_info, sync_broker

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


def register_broker_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    deps = ctx.protected_dependencies

    def _default_mode() -> str:
        return "paper" if ctx.settings.broker.paper_trading else "live"

    @app.get("/v1/broker/connection", dependencies=deps)
    async def broker_connection(mode: str | None = None) -> JSONResponse:
        return JSONResponse(content=connection_info(ctx.settings, mode or _default_mode()))

    @app.post("/v1/broker/sync", dependencies=deps)
    async def broker_sync_route(mode: str | None = None) -> JSONResponse:
        return JSONResponse(content=await sync_broker(ctx.settings, mode or _default_mode()))
