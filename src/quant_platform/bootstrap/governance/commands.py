"""Smoke and performance governance command composition."""

from __future__ import annotations

import traceback
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


def preflight_payload(
    settings: PlatformSettings,
    *,
    profile: str,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> tuple[dict[str, Any], bool]:
    from dataclasses import asdict

    from quant_platform.core.domain.production import ProductionProfile
    from quant_platform.services.governance_service.preflight import evaluate_preflight

    report = evaluate_preflight(
        settings,
        profile=ProductionProfile(profile),
        instrument_contracts=instrument_contracts,
    )
    return asdict(report), report.passed


async def smoke_command(settings: PlatformSettings) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.session.public_api import create_paper_session
    from quant_platform.core.contracts.common import BrokerHealthStatus
    from quant_platform.core.domain.production import ProductionProfile
    from quant_platform.services.governance_service.preflight import evaluate_preflight

    results: list[tuple[str, bool, str]] = []

    pg_dsn = settings.storage.postgres_dsn
    if pg_dsn:
        try:
            from sqlalchemy import text as sa_text
            from sqlalchemy.ext.asyncio import create_async_engine

            engine = create_async_engine(pg_dsn, echo=False)
            async with engine.connect() as conn:
                await conn.execute(sa_text("SELECT 1"))
            await engine.dispose()
            results.append(("postgres", True, "SELECT 1 OK"))
        except Exception as exc:
            results.append(("postgres", False, traceback.format_exc(limit=2).strip()))
            log.error("smoke.postgres_failed", error=str(exc))
    else:
        results.append(("postgres", True, "not configured - skipped"))

    redis_url = settings.storage.redis_url
    if redis_url:
        try:
            from quant_platform.bootstrap.persistence.redis_factory import create_async_redis_client

            client = create_async_redis_client(redis_url, socket_timeout=5.0)
            ok = await client.ping()
            await client.aclose()
            if ok:
                results.append(("redis", True, "PING OK"))
            else:
                results.append(("redis", False, "PING returned False"))
        except Exception as exc:
            results.append(("redis", False, traceback.format_exc(limit=2).strip()))
            log.error("smoke.redis_failed", error=str(exc))
    else:
        results.append(("redis", True, "not configured - skipped"))

    try:
        session = create_paper_session(settings, initial_cash=Decimal("0"))
        await session.account_broker.connect()
        health = await session.account_broker.health_check()
        await session.account_broker.disconnect()
        if health.status == BrokerHealthStatus.CONNECTED:
            results.append(("broker", True, f"CONNECTED latency={health.latency_ms}ms"))
        else:
            results.append(
                ("broker", False, f"status={health.status.value} detail={health.detail}")
            )
    except Exception as exc:
        results.append(("broker", False, traceback.format_exc(limit=2).strip()))
        log.error("smoke.broker_failed", error=str(exc))

    try:
        preflight = evaluate_preflight(
            settings,
            profile=ProductionProfile.PAPER,
            instrument_contracts={},
        )
        if preflight.passed:
            results.append(("preflight", True, f"all {len(preflight.checks)} checks passed"))
        else:
            failed = [c.name for c in preflight.failures]
            results.append(("preflight", False, f"failed checks: {failed}"))
    except Exception as exc:
        results.append(("preflight", False, traceback.format_exc(limit=2).strip()))
        log.error("smoke.preflight_failed", error=str(exc))

    all_passed = all(ok for _, ok, _ in results)
    return UseCaseResult(
        status=UseCaseStatus.OK if all_passed else UseCaseStatus.FAILED,
        payload={
            "smoke_passed": all_passed,
            "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in results],
        },
        exit_code=0 if all_passed else 1,
        presentation=ResultPresentation.JSON,
    )


async def performance_snapshot_command(
    settings: PlatformSettings,
    *,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    nav: Decimal,
    gross_exposure: Decimal,
    cash: Decimal,
    source: str,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.core.domain.production import NavSnapshot
    from quant_platform.infrastructure.performance import build_performance_repository

    repo = build_performance_repository(settings.storage.postgres_dsn)
    snapshot = NavSnapshot(
        snapshot_id=uuid.uuid4(),
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        net_asset_value=nav,
        gross_exposure=gross_exposure,
        cash=cash,
        source=source,
    )
    await repo.save_nav_snapshot(snapshot)
    return UseCaseResult(
        payload={"snapshot_id": str(snapshot.snapshot_id), "saved": True},
        presentation=ResultPresentation.JSON,
    )


async def performance_report_command(
    settings: PlatformSettings,
    *,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    window: int,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.infrastructure.performance import build_performance_repository

    repo = build_performance_repository(settings.storage.postgres_dsn)
    report = await repo.performance_report(strategy_run_id, as_of=as_of, window=window)
    return UseCaseResult(payload=vars(report), presentation=ResultPresentation.JSON)


async def performance_heartbeat_command(
    settings: PlatformSettings,
    *,
    component: str,
    as_of: datetime,
    status: str,
    detail: str,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.core.domain.production import RuntimeHeartbeat
    from quant_platform.infrastructure.performance import build_performance_repository

    repo = build_performance_repository(settings.storage.postgres_dsn)
    heartbeat = RuntimeHeartbeat(
        component=component,
        as_of=as_of,
        status=status,
        detail=detail,
    )
    await repo.save_runtime_heartbeat(heartbeat)
    return UseCaseResult(payload=vars(heartbeat), presentation=ResultPresentation.JSON)


__all__ = [
    "performance_heartbeat_command",
    "performance_report_command",
    "performance_snapshot_command",
    "preflight_payload",
    "smoke_command",
]
