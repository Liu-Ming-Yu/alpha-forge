"""Strategy state and engine evidence routes."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from quant_platform.bootstrap.operator_api.queries import list_strategy_runs_payload

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


async def strategy_lifecycle_response(
    ctx: OperatorApiRouteContext,
    strategy_run_id: uuid.UUID,
    engine_name: str = "cross_sectional_equity_v1",
    engine_version: str = "0.1.0",
    max_drawdown_limit: float = -0.15,
) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.current_strategy_lifecycle(
                strategy_run_id=strategy_run_id,
                engine_name=engine_name,
                engine_version=engine_version,
                max_drawdown_limit=max_drawdown_limit,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="lifecycle query timed out") from exc
    return JSONResponse(content=asdict(view))


async def regime_state_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.current_regime_state(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="regime query timed out") from exc
    data = asdict(view)
    data["as_of"] = str(data["as_of"])
    return JSONResponse(content=data)


async def strategy_runs_response(
    ctx: OperatorApiRouteContext,
    limit: int = 20,
    status: str | None = None,
) -> JSONResponse:
    try:
        rows = await asyncio.wait_for(
            list_strategy_runs_payload(
                ctx.settings,
                limit=limit,
                status_filter=status,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="strategy runs query timed out") from exc
    return JSONResponse(content={"runs": rows, "count": len(rows)})


async def signal_decay_response(
    ctx: OperatorApiRouteContext,
    engine_name: str = "cross_sectional_equity_v1",
    window: int = 500,
) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.current_signal_decay(engine_name=engine_name, window=window),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="signal decay query timed out") from exc
    data = asdict(view)
    data["as_of"] = str(data["as_of"])
    return JSONResponse(content=data)


async def engine_budgets_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    try:
        views = await asyncio.wait_for(
            ctx.builder.engine_budgets(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="engine budgets query timed out") from exc
    rows = []
    for view in views:
        row = asdict(view)
        for key in ("capital_weight", "max_gross", "max_turnover"):
            row[key] = float(row[key])
        rows.append(row)
    return JSONResponse(content={"budgets": rows})


async def combined_exposure_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.combined_exposure(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="exposure query timed out") from exc
    data = asdict(view)
    data["as_of"] = str(data["as_of"])
    for key in ("allocated_capital_weight", "reserved_cash_weight"):
        data[key] = float(data[key])
    return JSONResponse(content=data)


async def signal_contributions_response(
    ctx: OperatorApiRouteContext,
    strategy_run_id: uuid.UUID | None = None,
    instrument_id: uuid.UUID | None = None,
    limit: int = 500,
) -> JSONResponse:
    try:
        views = await asyncio.wait_for(
            ctx.builder.signal_contributions(
                strategy_run_id=strategy_run_id,
                instrument_id=instrument_id,
                limit=limit,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="signal contributions query timed out",
        ) from exc
    rows = []
    for view in views:
        row = asdict(view)
        row["score_id"] = str(row["score_id"])
        row["strategy_run_id"] = str(row["strategy_run_id"])
        row["instrument_id"] = str(row["instrument_id"])
        row["as_of"] = str(row["as_of"])
        rows.append(row)
    return JSONResponse(content={"contributions": rows})


def register_strategy_state_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    protected_dependencies = ctx.protected_dependencies

    @app.get("/strategy/lifecycle/{strategy_run_id}", dependencies=protected_dependencies)
    async def strategy_lifecycle(
        strategy_run_id: uuid.UUID,
        engine_name: str = "cross_sectional_equity_v1",
        engine_version: str = "0.1.0",
        max_drawdown_limit: float = -0.15,
    ) -> JSONResponse:
        return await strategy_lifecycle_response(
            ctx,
            strategy_run_id=strategy_run_id,
            engine_name=engine_name,
            engine_version=engine_version,
            max_drawdown_limit=max_drawdown_limit,
        )

    @app.get("/strategy/regime", dependencies=protected_dependencies)
    async def regime_state() -> JSONResponse:
        return await regime_state_response(ctx)

    @app.get("/strategy/runs", dependencies=protected_dependencies)
    async def strategy_runs(
        limit: int = 20,
        status: str | None = None,
    ) -> JSONResponse:
        return await strategy_runs_response(ctx, limit=limit, status=status)

    @app.get("/strategy/signal-decay", dependencies=protected_dependencies)
    async def signal_decay(
        engine_name: str = "cross_sectional_equity_v1",
        window: int = 500,
    ) -> JSONResponse:
        return await signal_decay_response(ctx, engine_name=engine_name, window=window)

    @app.get("/strategy/engines/budgets", dependencies=protected_dependencies)
    async def engine_budgets() -> JSONResponse:
        return await engine_budgets_response(ctx)

    @app.get("/strategy/engines/exposure", dependencies=protected_dependencies)
    async def combined_exposure() -> JSONResponse:
        return await combined_exposure_response(ctx)

    @app.get("/signals/contributions", dependencies=protected_dependencies)
    async def signal_contributions(
        strategy_run_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        limit: int = 500,
    ) -> JSONResponse:
        return await signal_contributions_response(
            ctx,
            strategy_run_id=strategy_run_id,
            instrument_id=instrument_id,
            limit=limit,
        )
