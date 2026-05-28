"""Governance evidence artifact command wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.bootstrap.governance.paper_soak import (
    build_data_health_evidence,
    order_latency_section,
    reconciliation_section,
)
from quant_platform.bootstrap.governance.repositories import build_performance_repository
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.persistence.postgres import create_pg_engine

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def paper_soak_report_command(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    strategy_run_id: uuid.UUID,
    contracts_file: str | None,
    signal_name: str,
    signal_type: str,
    bar_seconds: int,
    data_health_window_days: int,
    order_latency_window_days: int,
    output_path: Path,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.governance_service.paper_soak import (
        build_paper_soak_report,
        write_paper_soak_report,
    )

    await verify_postgres_schema(settings)
    performance_repo = build_performance_repository(settings.storage.postgres_dsn)
    reconciliation = await reconciliation_section(settings)
    order_latency = await order_latency_section(
        settings,
        as_of=as_of,
        window_days=order_latency_window_days,
    )
    payload = await build_paper_soak_report(
        settings,
        as_of=as_of,
        strategy_run_id=strategy_run_id,
        contracts_file=contracts_file,
        signal_name=signal_name,
        signal_type=signal_type,
        bar_seconds=bar_seconds,
        data_health_window_days=data_health_window_days,
        order_latency_window_days=order_latency_window_days,
        performance_repository=performance_repo,
        data_health_builder=build_data_health_evidence,
        signal_gate=performance_repo,
        reconciliation_evidence=reconciliation,
        order_latency_evidence=order_latency,
    )
    output = write_paper_soak_report(payload, output_path)
    failures = [
        section
        for section in ("broker_health", "lifecycle_result", "data_health", "signal_gate")
        if not bool(payload.get(section, {}).get("passed", False))
    ]
    if not bool(payload.get("nav_snapshot", {}).get("passed", False)):
        failures.append("nav_snapshot")
    reconciliation = payload.get("reconciliation") or {}
    if reconciliation.get("drift_detected") is True:
        failures.append("reconciliation")
    if not bool(payload.get("order_latency", {}).get("passed", False)):
        failures.append("order_latency")
    log.info(
        "paper_soak.report.written",
        output=str(output),
        failures=failures,
    )
    return UseCaseResult(
        payload={"path": str(output), "failures": failures, **payload},
        presentation=ResultPresentation.JSON,
    )


async def simulator_calibration_command(
    settings: PlatformSettings,
    *,
    subcommand: str,
    as_of: datetime,
    lookback_days: int,
    floor_bps: float,
    min_sample_count: int,
    p90_safety_margin: float,
    output_path: Path,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.governance_service.simulator_calibration import (
        compute_calibration_report,
        load_paper_fills_from_postgres,
        write_calibration_report,
    )

    if subcommand != "report":
        raise OperatorUsageError(f"unknown simulator-calibration command: {subcommand}")

    observations = await load_paper_fills_from_postgres(
        settings,
        as_of=as_of,
        lookback_days=lookback_days,
        pg_engine_factory=create_pg_engine,
    )
    report = compute_calibration_report(
        observations,
        as_of=as_of,
        floor_bps=floor_bps,
        min_sample_count=min_sample_count,
        p90_safety_margin=p90_safety_margin,
    )

    output = write_calibration_report(report, output_path)
    log.info(
        "simulator_calibration.report.written",
        output=str(output),
        sample_count=report.sample_count,
        insufficient_data=report.insufficient_data,
        recommended_bps=report.overall_recommended_bps,
    )
    return UseCaseResult(
        payload={
            "path": str(output),
            "sample_count": report.sample_count,
            "insufficient_data": report.insufficient_data,
            "overall_recommended_bps": report.overall_recommended_bps,
            "buckets": [
                {
                    "tactic": b.tactic,
                    "adv_bucket": b.adv_bucket,
                    "sample_count": b.sample_count,
                    "median_slippage_bps": b.median_slippage_bps,
                    "p90_slippage_bps": b.p90_slippage_bps,
                    "recommended_bps": b.recommended_bps,
                }
                for b in report.buckets
            ],
        },
        presentation=ResultPresentation.JSON,
    )


__all__ = ["paper_soak_report_command", "simulator_calibration_command"]
