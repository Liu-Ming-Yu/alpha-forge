"""Response builders for cash, order, audit, and control endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from quant_platform.core.events import (
    ComplianceViolationWarning,
    MarketBarIngested,
    UnmatchedFillEvent,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


def cash_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    view = ctx.builder.cash_status()
    data = asdict(view)
    for key in ("settled_cash", "unsettled_cash", "reserved_cash", "available_cash"):
        data[key] = float(data[key])
    data["as_of"] = str(data["as_of"])
    return JSONResponse(content=data)


async def blotter_response(
    ctx: OperatorApiRouteContext,
    strategy_run_id: uuid.UUID,
) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.blotter(strategy_run_id), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="blotter query timed out") from exc
    data = asdict(view)
    data["as_of"] = str(data["as_of"])
    for entry in data["entries"]:
        entry["order_id"] = str(entry["order_id"])
        entry["instrument_id"] = str(entry["instrument_id"])
        for field_name in ("avg_fill_price", "vwap_at_submission", "commission_paid"):
            if entry.get(field_name) is not None:
                entry[field_name] = float(entry[field_name])
    return JSONResponse(content=data)


async def paper_gate_metrics_response(
    ctx: OperatorApiRouteContext,
    strategy_run_id: uuid.UUID,
) -> JSONResponse:
    try:
        view = await asyncio.wait_for(
            ctx.builder.paper_gate_metrics(strategy_run_id),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="metrics query timed out") from exc
    data = asdict(view)
    data["as_of"] = str(data["as_of"])
    for key in ("reject_rate", "broker_error_rate"):
        data[key] = float(data[key])
    if data["average_fill_slippage_bps"] is not None:
        data["average_fill_slippage_bps"] = float(data["average_fill_slippage_bps"])
    return JSONResponse(content=data)


async def audit_events_response(
    ctx: OperatorApiRouteContext,
    event_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    if ctx.selected_audit_sink is None or not hasattr(ctx.selected_audit_sink, "list_events"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="audit sink does not support list_events",
        )
    try:
        rows = await asyncio.wait_for(
            ctx.selected_audit_sink.list_events(
                event_type=event_type,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="audit query timed out",
        ) from exc
    return JSONResponse(content={"entries": rows, "count": len(rows)})


async def order_allocations_response(
    ctx: OperatorApiRouteContext,
    order_id: uuid.UUID,
) -> JSONResponse:
    try:
        views = await asyncio.wait_for(
            ctx.builder.order_allocations(order_id), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="allocations query timed out") from exc
    rows = []
    for view in views:
        row = asdict(view)
        row["order_id"] = str(row["order_id"])
        row["strategy_run_id"] = str(row["strategy_run_id"])
        row["instrument_id"] = str(row["instrument_id"])
        row["allocated_weight"] = float(row["allocated_weight"])
        if row["allocated_notional"] is not None:
            row["allocated_notional"] = float(row["allocated_notional"])
        rows.append(row)
    return JSONResponse(content={"allocations": rows})


async def compliance_violations_response(
    ctx: OperatorApiRouteContext,
    since_hours: int = 24,
    limit: int = 100,
) -> JSONResponse:
    try:
        history = await asyncio.wait_for(
            ctx.builder._event_history(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="compliance violations query timed out",
        ) from exc

    cutoff = (
        ctx.clock.now().replace(tzinfo=UTC) if ctx.clock.now().tzinfo is None else ctx.clock.now()
    )
    cutoff = cutoff - timedelta(hours=since_hours)
    rows = [
        {
            "order_id": str(event.order_id),
            "rule": event.rule,
            "detail": event.detail,
            "occurred_at": event.occurred_at.isoformat(),
        }
        for event in history
        if isinstance(event, ComplianceViolationWarning) and event.occurred_at >= cutoff
    ][-limit:]
    return JSONResponse(content={"violations": rows, "count": len(rows)})


async def unmatched_fills_response(
    ctx: OperatorApiRouteContext,
    limit: int = 100,
) -> JSONResponse:
    try:
        history = await asyncio.wait_for(
            ctx.builder._event_history(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="unmatched fills query timed out") from exc

    rows = [
        {
            "ib_order_id": event.ib_order_id,
            "exec_id": event.exec_id,
            "con_id": event.con_id,
            "occurred_at": event.occurred_at.isoformat(),
        }
        for event in history
        if isinstance(event, UnmatchedFillEvent)
    ][-limit:]
    return JSONResponse(content={"unmatched_fills": rows, "count": len(rows)})


def cash_ledger_detail_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    view = ctx.builder.cash_status()
    pending_lots = []
    for lot in getattr(ctx.ledger, "_unsettled_buys", []):
        pending_lots.append(
            {
                "lot_id": str(getattr(lot, "lot_id", "")),
                "quantity": int(getattr(lot, "quantity", 0)),
                "cost": float(getattr(lot, "cost", 0)),
                "settlement_date": str(getattr(lot, "settlement_date", "")),
            }
        )
    return JSONResponse(
        content={
            "as_of": str(view.as_of),
            "settled_cash": float(view.settled_cash),
            "unsettled_cash": float(view.unsettled_cash),
            "reserved_cash": float(view.reserved_cash),
            "available_cash": float(view.available_cash),
            "pending_lots_count": len(pending_lots),
            "pending_lots": pending_lots,
        }
    )


async def clear_kill_switch_response(
    ctx: OperatorApiRouteContext,
    reason: str = "operator-cleared via API",
    confirmation: str = "",
    body: dict[str, Any] | None = None,
) -> JSONResponse:
    if ctx.kill_switch_store is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="kill switch store not wired into the operator API",
        )
    if body:
        reason = str(body.get("reason", reason))
        confirmation = str(body.get("confirmation", confirmation))
    if not reason.strip():
        raise HTTPException(status_code=400, detail="reason is required")
    if confirmation != "CLEAR KILL SWITCH":
        raise HTTPException(
            status_code=400,
            detail='confirmation must equal "CLEAR KILL SWITCH"',
        )
    try:
        await asyncio.wait_for(
            ctx.kill_switch_store.clear(operator_id=reason, as_of=ctx.clock.now()),
            timeout=ctx.repo_timeout_seconds,
        )
        ctx.throttle_policy.clear_kill_switch(reason)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="kill switch clear timed out") from exc
    return JSONResponse(content={"status": "cleared"})


async def data_freshness_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    try:
        history = await asyncio.wait_for(
            ctx.builder._event_history(), timeout=ctx.repo_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="data freshness query timed out") from exc

    latest: dict[str, str] = {}
    for event in history:
        if isinstance(event, MarketBarIngested):
            instrument_id = str(event.instrument_id)
            timestamp = event.occurred_at.isoformat()
            if instrument_id not in latest or timestamp > latest[instrument_id]:
                latest[instrument_id] = timestamp

    rows = [
        {"instrument_id": instrument_id, "last_bar_at": timestamp}
        for instrument_id, timestamp in sorted(latest.items())
    ]
    return JSONResponse(content={"instruments": rows, "count": len(rows)})
