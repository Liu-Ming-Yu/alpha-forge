"""Operator API query and telemetry adapter facade.

FastAPI handlers should not construct database engines or import concrete
governance/research adapters directly.  This module keeps those outer-edge
dependencies in bootstrap while returning plain JSON-ready payloads.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.bootstrap.data.v2_datasets import build_dataset_catalog
from quant_platform.bootstrap.governance.repositories import build_performance_repository
from quant_platform.bootstrap.persistence.migrations import alembic_packaged_head
from quant_platform.core.domain.production import ProductionProfile

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def record_operator_http_request(
    *,
    method: str,
    endpoint: str,
    status: int,
    duration_seconds: float,
) -> None:
    from quant_platform.telemetry.metrics import record_http_request

    record_http_request(
        method=method,
        endpoint=endpoint,
        status=status,
        duration_seconds=duration_seconds,
    )


def render_operator_metrics() -> tuple[bytes, str]:
    from quant_platform.telemetry.metrics import render_latest

    return render_latest()


async def authorize_operator_viewer_api_key(
    *,
    raw_key: str,
    repository: Any,
    as_of: datetime,
) -> object | None:
    from quant_platform.services.governance_service.support.operator_auth import (
        authorize_operator_api_key,
    )

    return await authorize_operator_api_key(
        raw_key=raw_key,
        repository=repository,
        min_role="viewer",
        as_of=as_of,
    )


async def list_strategy_runs_payload(
    settings: PlatformSettings,
    *,
    limit: int,
    status_filter: str | None,
) -> list[dict[str, Any]]:
    if not settings.storage.postgres_dsn:
        return []

    from sqlalchemy import text

    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    params: dict[str, Any] = {"limit": max(1, min(limit, 500))}
    if status_filter:
        params["status"] = status_filter
        query = text("""
        SELECT run_id, strategy_name, strategy_version, run_type, status,
               config_snapshot, created_at, started_at, finished_at
        FROM strategy_runs
        WHERE status = :status
        ORDER BY COALESCE(started_at, created_at) DESC
        LIMIT :limit
        """)
    else:
        query = text("""
        SELECT run_id, strategy_name, strategy_version, run_type, status,
               config_snapshot, created_at, started_at, finished_at
        FROM strategy_runs
        ORDER BY COALESCE(started_at, created_at) DESC
        LIMIT :limit
        """)
    engine = create_pg_engine(settings.storage.postgres_dsn)
    async with engine.connect() as conn:
        rows = (await conn.execute(query, params)).mappings().all()
    return [_serialise_strategy_run(dict(row)) for row in rows]


async def postgres_ready(settings: PlatformSettings) -> bool:
    if not settings.storage.postgres_dsn:
        return True
    from sqlalchemy import text

    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    engine = create_pg_engine(settings.storage.postgres_dsn)
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def latest_feature_vector_age_seconds(settings: PlatformSettings) -> float | None:
    if not settings.storage.postgres_dsn:
        return None
    from sqlalchemy import text as sa_text

    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    engine = create_pg_engine(settings.storage.postgres_dsn)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                sa_text(
                    "SELECT EXTRACT(EPOCH FROM (now() - MAX(created_at))) "
                    "AS age_seconds FROM feature_vectors"
                )
            )
        ).first()
    return float(row[0]) if row and row[0] is not None else None


async def latest_readiness_snapshot_payload(
    settings: PlatformSettings,
    *,
    profile: str,
) -> dict[str, Any] | None:
    if not (settings.v2.enabled and settings.storage.postgres_dsn):
        return None
    try:
        from quant_platform.infrastructure.v2.postgres import build_v2_repository_bundle
    except ImportError:
        return None
    try:
        bundle = build_v2_repository_bundle(settings, require_postgres=True)
    except Exception:
        return None
    snapshot = await bundle.production_evidence.latest_readiness_snapshot(profile)
    if snapshot is None:
        return None
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "profile": snapshot.profile.value,
        "generated_at": snapshot.generated_at.isoformat(),
        "state": snapshot.state.value,
        "passed": snapshot.passed,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "severity": check.severity,
            }
            for check in snapshot.checks
        ],
    }


async def production_candidate_payload_for_api(
    settings: PlatformSettings,
    *,
    profile: str,
    as_of: datetime,
    clean_live_days: int,
) -> dict[str, Any]:
    from quant_platform.services.governance_service.production_candidate import (
        build_production_candidate_report,
        production_candidate_payload,
    )

    try:
        parsed_profile = ProductionProfile(profile)
    except ValueError as exc:
        raise ValueError(f"invalid profile: {profile!r}") from exc

    as_of = as_of.replace(tzinfo=UTC) if as_of.tzinfo is None else as_of.astimezone(UTC)
    evidence_repo = build_performance_repository(settings.storage.postgres_dsn)
    dataset_catalog = (
        build_dataset_catalog(settings.storage.postgres_dsn)
        if settings.storage.postgres_dsn and settings.v2.require_dataset_quorum
        else None
    )

    report = await build_production_candidate_report(
        settings,
        profile=parsed_profile,
        as_of=as_of,
        instrument_contracts={},
        clean_live_days=max(0, clean_live_days),
        evidence_repository=evidence_repo,
        dataset_catalog=dataset_catalog,
        packaged_migration_head=alembic_packaged_head(),
    )
    return production_candidate_payload(report)


def read_backtest_ic_report_payload(
    settings: PlatformSettings,
    *,
    run_id: str,
) -> tuple[int, dict[str, Any]]:
    from quant_platform.services.research_service.reports.ic_report import read_ic_report

    try:
        rid = uuid.UUID(run_id)
    except ValueError:
        return 400, {"detail": f"invalid run_id: {run_id!r}"}

    path = Path(settings.storage.object_store_root) / "backtest" / str(rid) / "ic_report.json"
    if not path.is_file():
        return 404, {"detail": f"no ic_report.json for run_id={rid}"}
    return 200, read_ic_report(path)


def _serialise_strategy_run(row: dict[str, Any]) -> dict[str, Any]:
    serialised = dict(row)
    for key in ("run_id",):
        if serialised.get(key) is not None:
            serialised[key] = str(serialised[key])
    for key in ("created_at", "started_at", "finished_at"):
        value = serialised.get(key)
        if value is not None and hasattr(value, "isoformat"):
            serialised[key] = value.isoformat()
        elif value is not None:
            serialised[key] = str(value)
    return serialised
