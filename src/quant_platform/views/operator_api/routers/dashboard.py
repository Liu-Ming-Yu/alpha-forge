"""Aggregated operator dashboard route."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from fastapi.responses import JSONResponse

from quant_platform.bootstrap.operator_api.queries import list_strategy_runs_payload
from quant_platform.views.operator_api.routers.cash_orders_audit import (
    audit_events_response,
    blotter_response,
    cash_response,
    compliance_violations_response,
    data_freshness_response,
    paper_gate_metrics_response,
)
from quant_platform.views.operator_api.routers.health import health_ready_response
from quant_platform.views.operator_api.routers.research_evidence import (
    paper_soak_latest_response,
    promotion_candidate_response,
    readiness_latest_response,
    research_campaigns_response,
    research_feature_audits_response,
)
from quant_platform.views.operator_api.routers.strategy_state import (
    combined_exposure_response,
    engine_budgets_response,
    regime_state_response,
    strategy_lifecycle_response,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


async def dashboard_summary_response(
    ctx: OperatorApiRouteContext,
    strategy_run_id: uuid.UUID | None = None,
) -> JSONResponse:
    runs: list[dict[str, Any]] = []
    selected_run_id = strategy_run_id
    try:
        runs = await asyncio.wait_for(
            list_strategy_runs_payload(
                ctx.settings,
                limit=10,
                status_filter=None,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError:
        runs = []
    if selected_run_id is None and runs:
        selected_run_id = uuid.UUID(str(runs[0]["run_id"]))

    async def _safe(name: str, factory: Callable[[], Awaitable[object]]) -> object:
        try:
            return await asyncio.wait_for(factory(), timeout=ctx.repo_timeout_seconds)
        except Exception as exc:
            return {"error": f"{name} unavailable: {exc}"}

    health_view = await _safe("health", ctx.builder.broker_health)
    if not isinstance(health_view, dict):
        health = asdict(cast("Any", health_view))
        last_hb = health.get("last_heartbeat_at")
        if last_hb is not None and hasattr(last_hb, "isoformat"):
            health["last_heartbeat_at"] = last_hb.isoformat()
    else:
        health = health_view

    cash_payload = ctx.json_payload(cash_response(ctx))

    ready = ctx.json_payload(await health_ready_response(ctx))
    regime = ctx.json_payload(await regime_state_response(ctx))
    budgets = ctx.json_payload(await engine_budgets_response(ctx))
    exposure = ctx.json_payload(await combined_exposure_response(ctx))
    freshness = ctx.json_payload(await data_freshness_response(ctx))
    audit = ctx.json_payload(await audit_events_response(ctx, limit=10))
    compliance = ctx.json_payload(await compliance_violations_response(ctx, limit=10))
    research_campaign_payload = ctx.json_payload(await research_campaigns_response(ctx, limit=10))
    feature_audit_payload = ctx.json_payload(await research_feature_audits_response(ctx, limit=10))
    forecast_sources = [
        source
        for source, weight in ctx.settings.alpha.source_weights.items()
        if source != "classical" and float(weight) > 0
    ]
    forecast_evidence_payload: object = await _safe(
        "forecast evidence",
        lambda: ctx.builder.forecast_evidence(
            forecast_sources,
            stale_after_hours=ctx.settings.production.prediction_evidence_stale_after_hours,
            min_confidence=ctx.settings.production.prediction_evidence_min_confidence,
        ),
    )
    if not isinstance(forecast_evidence_payload, dict):
        forecast_evidence_payload = [
            {
                **asdict(cast("Any", row)),
                "as_of": row.as_of.isoformat(),
                "latest_prediction_at": (
                    row.latest_prediction_at.isoformat()
                    if row.latest_prediction_at is not None
                    else None
                ),
            }
            for row in cast("Any", forecast_evidence_payload)
        ]

    promotion_candidate_payload: dict[str, Any] | None = None
    try:
        promotion_candidate_payload = ctx.json_payload(
            await asyncio.wait_for(
                promotion_candidate_response(ctx, profile="paper", clean_live_days=0),
                timeout=ctx.repo_timeout_seconds,
            )
        )
    except Exception as exc:
        promotion_candidate_payload = {"error": f"promotion-candidate unavailable: {exc}"}

    readiness_snapshot_payload: dict[str, Any] | None = None
    try:
        readiness_snapshot_payload = ctx.json_payload(
            await asyncio.wait_for(
                readiness_latest_response(ctx, profile="paper"),
                timeout=ctx.repo_timeout_seconds,
            )
        )
    except Exception as exc:
        readiness_snapshot_payload = {"error": f"readiness snapshot unavailable: {exc}"}

    paper_soak_payload: dict[str, Any] | None = None
    try:
        paper_soak_payload = ctx.json_payload(
            await asyncio.wait_for(
                paper_soak_latest_response(ctx),
                timeout=ctx.repo_timeout_seconds,
            )
        )
    except Exception as exc:
        paper_soak_payload = {"error": f"paper-soak metadata unavailable: {exc}"}

    run_payload: dict[str, Any] = {}
    if selected_run_id is not None:
        run_payload["strategy_run_id"] = str(selected_run_id)
        run_payload["blotter"] = ctx.json_payload(await blotter_response(ctx, selected_run_id))
        run_payload["metrics"] = ctx.json_payload(
            await paper_gate_metrics_response(ctx, selected_run_id)
        )
        run_payload["lifecycle"] = ctx.json_payload(
            await strategy_lifecycle_response(ctx, selected_run_id)
        )

    kill_switch_state = None
    if ctx.kill_switch_store is not None:
        try:
            state = await asyncio.wait_for(
                ctx.kill_switch_store.get(),
                timeout=ctx.repo_timeout_seconds,
            )
            kill_switch_state = asdict(state)
            for key in ("activated_at", "cleared_at"):
                value = kill_switch_state.get(key)
                if value is not None and hasattr(value, "isoformat"):
                    kill_switch_state[key] = value.isoformat()
        except Exception as exc:
            kill_switch_state = {"error": f"kill switch unavailable: {exc}"}

    return JSONResponse(
        content={
            "as_of": ctx.clock.now().isoformat(),
            "capabilities": ctx.capabilities_payload(),
            "ready": ready,
            "health": health,
            "cash": cash_payload,
            "regime": regime,
            "strategy_runs": {"runs": runs, "count": len(runs)},
            "selected_run": run_payload,
            "engines": {"budgets": budgets.get("budgets", []), "exposure": exposure},
            "freshness": freshness,
            "research_campaigns": research_campaign_payload,
            "feature_audits": feature_audit_payload,
            "forecast_evidence": forecast_evidence_payload,
            "audit": audit,
            "compliance": compliance,
            "kill_switch": kill_switch_state,
            "production_candidate": promotion_candidate_payload,
            "readiness_snapshot": readiness_snapshot_payload,
            "paper_soak": paper_soak_payload,
        }
    )


def register_dashboard_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    protected_dependencies = ctx.protected_dependencies

    @app.get("/dashboard/summary", dependencies=protected_dependencies)
    async def dashboard_summary(strategy_run_id: uuid.UUID | None = None) -> JSONResponse:
        return await dashboard_summary_response(ctx, strategy_run_id=strategy_run_id)
