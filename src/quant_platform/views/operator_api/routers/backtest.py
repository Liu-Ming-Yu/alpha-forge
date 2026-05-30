"""Backtest evidence routes — list runs and build graphable results.

Read-only: these serve persisted walk-forward evidence as equity/drawdown/IC
series for the console's live backtest graphs. Running a *fresh* backtest goes
through the command runner (``research-campaign run``), gated by the execution
opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from quant_platform.views.operator_api.backtest import (
    list_backtest_runs,
    load_backtest_result,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


def register_backtest_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    deps = ctx.protected_dependencies

    @app.get("/v1/backtest/runs", dependencies=deps)
    async def backtest_runs() -> JSONResponse:
        return JSONResponse(content={"runs": list_backtest_runs(ctx.settings)})

    @app.get("/v1/backtest/result", dependencies=deps)
    async def backtest_result(run_id: str) -> JSONResponse:
        result = load_backtest_result(ctx.settings, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="backtest run not found")
        return JSONResponse(content=result)
