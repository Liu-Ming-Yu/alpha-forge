"""Cash, order, audit, and operational-control routes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from quant_platform.views.operator_api.routers.cash_orders_audit_responses import (
    audit_events_response,
    blotter_response,
    cash_ledger_detail_response,
    cash_response,
    clear_kill_switch_response,
    compliance_violations_response,
    data_freshness_response,
    order_allocations_response,
    paper_gate_metrics_response,
    unmatched_fills_response,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext

__all__ = [
    "audit_events_response",
    "blotter_response",
    "cash_response",
    "compliance_violations_response",
    "data_freshness_response",
    "paper_gate_metrics_response",
    "register_cash_orders_audit_routes",
]


def register_cash_orders_audit_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    protected_dependencies = ctx.protected_dependencies

    @app.get("/cash", dependencies=protected_dependencies)
    async def cash() -> JSONResponse:
        return cash_response(ctx)

    @app.get("/blotter/{strategy_run_id}", dependencies=protected_dependencies)
    async def blotter(strategy_run_id: uuid.UUID) -> JSONResponse:
        return await blotter_response(ctx, strategy_run_id)

    @app.get("/metrics/{strategy_run_id}", dependencies=protected_dependencies)
    async def metrics(strategy_run_id: uuid.UUID) -> JSONResponse:
        return await paper_gate_metrics_response(ctx, strategy_run_id)

    @app.get("/audit", dependencies=protected_dependencies)
    async def audit_events(
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> JSONResponse:
        return await audit_events_response(
            ctx,
            event_type=event_type,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    @app.get("/orders/{order_id}/allocations", dependencies=protected_dependencies)
    async def order_allocations(order_id: uuid.UUID) -> JSONResponse:
        return await order_allocations_response(ctx, order_id)

    @app.get("/v1/compliance/violations", dependencies=protected_dependencies)
    async def compliance_violations(
        since_hours: int = 24,
        limit: int = 100,
    ) -> JSONResponse:
        return await compliance_violations_response(
            ctx,
            since_hours=since_hours,
            limit=limit,
        )

    @app.get("/v1/fills/unmatched", dependencies=protected_dependencies)
    async def unmatched_fills(limit: int = 100) -> JSONResponse:
        return await unmatched_fills_response(ctx, limit=limit)

    @app.get("/v1/cash/ledger", dependencies=protected_dependencies)
    async def cash_ledger_detail() -> JSONResponse:
        return cash_ledger_detail_response(ctx)

    @app.post("/v1/kill-switch/clear", dependencies=protected_dependencies)
    async def clear_kill_switch_endpoint(
        reason: str = "operator-cleared via API",
        confirmation: str = "",
        body: dict[str, Any] | None = None,
    ) -> JSONResponse:
        return await clear_kill_switch_response(
            ctx,
            reason=reason,
            confirmation=confirmation,
            body=body,
        )

    @app.get("/v1/data/freshness", dependencies=protected_dependencies)
    async def data_freshness() -> JSONResponse:
        return await data_freshness_response(ctx)
