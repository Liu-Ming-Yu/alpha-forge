"""Typed governance request dispatch helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.bootstrap.operator_adapters.common import default_paper_soak_dir

if TYPE_CHECKING:
    from quant_platform.application.governance import (
        DatasetQuorumRequest,
        PaperSoakReportRequest,
        PerformanceRequest,
        SignalGateRequest,
        SimulatorCalibrationRequest,
        TextGateRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def run_performance_request(
    settings: PlatformSettings,
    request: PerformanceRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.governance import (
        PerformanceHeartbeatRequest,
        PerformanceReportRequest,
        PerformanceSnapshotRequest,
    )
    from quant_platform.bootstrap.governance.commands import (
        performance_heartbeat_command,
        performance_report_command,
        performance_snapshot_command,
    )

    if isinstance(request, PerformanceSnapshotRequest):
        return await performance_snapshot_command(
            settings,
            strategy_run_id=request.strategy_run_id,
            as_of=request.as_of,
            nav=request.nav,
            gross_exposure=request.gross_exposure,
            cash=request.cash,
            source=request.source,
        )
    if isinstance(request, PerformanceReportRequest):
        return await performance_report_command(
            settings,
            strategy_run_id=request.strategy_run_id,
            as_of=request.as_of,
            window=request.window,
        )
    if isinstance(request, PerformanceHeartbeatRequest):
        return await performance_heartbeat_command(
            settings,
            component=request.component,
            as_of=request.as_of,
            status=request.status,
            detail=request.detail,
        )
    raise TypeError(f"unknown performance request type: {type(request).__name__}")


async def run_signal_gate_request(
    settings: PlatformSettings,
    request: SignalGateRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.governance.gates import signal_gate_command

    return await signal_gate_command(
        settings,
        subcommand=request.command,
        signal_name=request.signal_name,
        signal_type=request.signal_type,
        as_of=request.as_of,
        daily_ic=request.daily_ic,
        observations=request.observations,
        drawdown=request.drawdown,
        turnover=request.turnover,
    )


async def run_text_gate_request(
    settings: PlatformSettings, request: TextGateRequest
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.governance.gates import text_gate_command

    return await text_gate_command(
        settings,
        subcommand=request.command,
        strategy_name=request.strategy_name,
        as_of=request.as_of,
        daily_ic=request.daily_ic,
        observations=request.observations,
    )


async def run_paper_soak_request(
    settings: PlatformSettings,
    request: PaperSoakReportRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.governance.evidence import paper_soak_report_command

    output_path = request.output
    if output_path is None:
        canonical_dir = default_paper_soak_dir(settings)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        slug = request.as_of.strftime("%Y%m%dT%H%M%SZ")
        output_path = canonical_dir / f"paper_soak_{slug}.json"
    return await paper_soak_report_command(
        settings,
        as_of=request.as_of,
        strategy_run_id=request.strategy_run_id,
        contracts_file=request.contracts_file,
        signal_name=request.signal_name,
        signal_type=request.signal_type,
        bar_seconds=request.bar_seconds,
        data_health_window_days=request.data_health_window_days,
        order_latency_window_days=request.order_latency_window_days,
        output_path=output_path,
    )


async def run_simulator_calibration_request(
    settings: PlatformSettings,
    request: SimulatorCalibrationRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.governance.evidence import simulator_calibration_command

    output_path = request.output
    if output_path is None:
        canonical_dir = Path(settings.storage.object_store_root) / "calibration"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        slug = request.as_of.strftime("%Y%m%dT%H%M%SZ")
        output_path = canonical_dir / f"simulator_calibration_{slug}.json"
    return await simulator_calibration_command(
        settings,
        subcommand="report",
        as_of=request.as_of,
        lookback_days=request.lookback_days,
        floor_bps=request.floor_bps,
        min_sample_count=request.min_sample_count,
        p90_safety_margin=request.p90_safety_margin,
        output_path=output_path,
    )


async def run_dataset_quorum_request(
    settings: PlatformSettings,
    request: DatasetQuorumRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.bootstrap.governance.dataset import dataset_quorum_command

    return await dataset_quorum_command(
        settings,
        subcommand=request.command,
        dataset_kind=request.dataset_kind,
        as_of=request.as_of,
        vendor_bars=request.vendor_bars,
        required_vendor_count=request.required_vendor_count,
        max_disagreement_bps=request.max_disagreement_bps,
    )


__all__ = [
    "run_dataset_quorum_request",
    "run_paper_soak_request",
    "run_performance_request",
    "run_signal_gate_request",
    "run_simulator_calibration_request",
    "run_text_gate_request",
]
