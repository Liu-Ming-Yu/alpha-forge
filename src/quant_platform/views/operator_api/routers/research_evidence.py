"""Research, governance evidence, readiness, and artifact routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from quant_platform.bootstrap.operator_api.queries import (
    latest_readiness_snapshot_payload,
    production_candidate_payload_for_api,
    read_backtest_ic_report_payload,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext


async def research_campaigns_response(
    ctx: OperatorApiRouteContext,
    limit: int = 20,
) -> JSONResponse:
    try:
        payload = await asyncio.wait_for(
            ctx.list_research_campaigns(limit=max(1, min(limit, 100))),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="research campaigns query timed out",
        ) from exc
    return JSONResponse(content=payload)


async def research_campaign_detail_response(
    ctx: OperatorApiRouteContext,
    run_id: str,
) -> JSONResponse:
    try:
        payload = await asyncio.wait_for(
            ctx.read_research_campaign(run_id),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="research campaign query timed out",
        ) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail=f"campaign not found: {run_id}")
    return JSONResponse(content=payload)


async def research_feature_audits_response(
    ctx: OperatorApiRouteContext,
    feature_name: str | None = None,
    limit: int = 50,
) -> JSONResponse:
    try:
        payload = await asyncio.wait_for(
            ctx.list_feature_audits(
                feature_name=feature_name,
                limit=limit,
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="feature audits query timed out") from exc
    return JSONResponse(content=payload)


async def paper_soak_latest_response(ctx: OperatorApiRouteContext) -> JSONResponse:
    """Return metadata for the most recent persisted paper-soak report."""

    root = Path(ctx.settings.storage.object_store_root) / "paper_soak"
    if not root.is_dir():
        return JSONResponse(content={"path": None, "passed_sections": {}, "generated_at": None})

    candidates = sorted(
        (path for path in root.glob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return JSONResponse(content={"path": None, "passed_sections": {}, "generated_at": None})

    latest = candidates[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={"path": str(latest), "error": f"unreadable soak report: {exc}"},
        )
    passed: dict[str, bool] = {}
    for key in (
        "broker_health",
        "lifecycle_result",
        "nav_snapshot",
        "data_health",
        "signal_gate",
        "order_latency",
    ):
        section = payload.get(key)
        if isinstance(section, dict):
            passed[key] = bool(section.get("passed", False))
    reconciliation = payload.get("reconciliation")
    if isinstance(reconciliation, dict):
        passed["reconciliation"] = not bool(reconciliation.get("drift_detected", False))
    return JSONResponse(
        content={
            "path": str(latest),
            "generated_at": payload.get("generated_at"),
            "passed_sections": passed,
            "version": payload.get("version"),
        }
    )


async def readiness_latest_response(
    ctx: OperatorApiRouteContext,
    profile: str = "paper",
) -> JSONResponse:
    try:
        payload = await asyncio.wait_for(
            latest_readiness_snapshot_payload(ctx.settings, profile=profile),
            timeout=ctx.repo_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="readiness query timed out") from exc
    if payload is None:
        return JSONResponse(content={"profile": profile, "snapshot": None})
    return JSONResponse(content={"profile": profile, "snapshot": payload})


async def promotion_candidate_response(
    ctx: OperatorApiRouteContext,
    profile: str = "paper",
    clean_live_days: int = 0,
) -> JSONResponse:
    try:
        payload = await asyncio.wait_for(
            production_candidate_payload_for_api(
                ctx.settings,
                profile=profile,
                as_of=ctx.clock.now(),
                clean_live_days=max(0, clean_live_days),
            ),
            timeout=ctx.repo_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="production-candidate query timed out",
        ) from exc

    return JSONResponse(content=payload)


def research_ic_response(ctx: OperatorApiRouteContext, run_id: str) -> JSONResponse:
    status_code, payload = read_backtest_ic_report_payload(ctx.settings, run_id=run_id)
    return JSONResponse(status_code=status_code, content=payload)


def register_research_evidence_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    protected_dependencies = ctx.protected_dependencies

    @app.get("/research/campaigns", dependencies=protected_dependencies)
    async def research_campaigns(limit: int = 20) -> JSONResponse:
        return await research_campaigns_response(ctx, limit=limit)

    @app.get("/research/campaigns/{run_id}", dependencies=protected_dependencies)
    async def research_campaign_detail(run_id: str) -> JSONResponse:
        return await research_campaign_detail_response(ctx, run_id=run_id)

    @app.get("/research/features/audits", dependencies=protected_dependencies)
    async def research_feature_audits(
        feature_name: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        return await research_feature_audits_response(
            ctx,
            feature_name=feature_name,
            limit=limit,
        )

    @app.get("/v1/paper-soak/latest", dependencies=protected_dependencies)
    async def paper_soak_latest() -> JSONResponse:
        return await paper_soak_latest_response(ctx)

    @app.get("/v1/readiness/latest", dependencies=protected_dependencies)
    async def readiness_latest(profile: str = "paper") -> JSONResponse:
        return await readiness_latest_response(ctx, profile=profile)

    @app.get("/v1/promotion/candidate", dependencies=protected_dependencies)
    async def promotion_candidate(
        profile: str = "paper",
        clean_live_days: int = 0,
    ) -> JSONResponse:
        return await promotion_candidate_response(
            ctx,
            profile=profile,
            clean_live_days=clean_live_days,
        )

    @app.get("/research/ic/{run_id}", dependencies=protected_dependencies)
    async def research_ic(run_id: str) -> JSONResponse:
        return research_ic_response(ctx, run_id=run_id)
