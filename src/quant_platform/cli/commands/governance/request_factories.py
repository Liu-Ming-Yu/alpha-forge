"""Governance CLI Namespace to typed request mappings."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.governance import (
    DatasetQuorumRequest,
    PaperSoakReportRequest,
    PerformanceHeartbeatRequest,
    PerformanceReportRequest,
    PerformanceSnapshotRequest,
    ProductionCandidateRequest,
    ReadinessRequest,
    SignalGateRequest,
    SimulatorCalibrationRequest,
    TextGateRequest,
)

if TYPE_CHECKING:
    import argparse


def performance_snapshot_request(args: argparse.Namespace) -> PerformanceSnapshotRequest:
    return PerformanceSnapshotRequest(
        strategy_run_id=args.strategy_run_id,
        as_of=args.as_of,
        nav=args.nav,
        gross_exposure=args.gross_exposure,
        cash=args.cash,
        source=args.source,
    )


def performance_report_request(args: argparse.Namespace) -> PerformanceReportRequest:
    return PerformanceReportRequest(
        strategy_run_id=args.strategy_run_id,
        as_of=args.as_of,
        window=args.window,
    )


def performance_heartbeat_request(args: argparse.Namespace) -> PerformanceHeartbeatRequest:
    return PerformanceHeartbeatRequest(
        component=args.component,
        as_of=args.as_of,
        status=args.status,
        detail=args.detail,
    )


def signal_gate_request(args: argparse.Namespace) -> SignalGateRequest:
    return SignalGateRequest(
        command=args.signal_gate_command,
        signal_name=args.signal_name,
        signal_type=args.signal_type,
        as_of=args.as_of,
        daily_ic=getattr(args, "daily_ic", None),
        observations=getattr(args, "observations", 1),
        drawdown=getattr(args, "drawdown", 0.0),
        turnover=getattr(args, "turnover", 0.0),
    )


def text_gate_request(args: argparse.Namespace) -> TextGateRequest:
    return TextGateRequest(
        command=args.text_gate_command,
        strategy_name=args.strategy_name,
        as_of=args.as_of,
        daily_ic=getattr(args, "daily_ic", None),
        observations=getattr(args, "observations", 1),
    )


def readiness_request(args: argparse.Namespace) -> ReadinessRequest:
    return ReadinessRequest(
        command=args.readiness_command,
        profile=args.profile,
        as_of=args.as_of,
        contracts_file=args.contracts_file,
        component=args.component,
        soak_report=args.soak_report,
        backup_manifest=args.backup_manifest,
        signal_name=args.signal_name,
        signal_type=args.signal_type,
        check_broker=args.check_broker,
    )


def production_candidate_request(args: argparse.Namespace) -> ProductionCandidateRequest:
    return ProductionCandidateRequest(
        command=args.production_candidate_command,
        profile=args.profile,
        as_of=args.as_of,
        contracts_file=args.contracts_file,
        component=args.component,
        soak_report=args.soak_report,
        backup_manifest=args.backup_manifest,
        signal_sources=tuple(args.signal_source or ()),
        primary_signal_name=args.primary_signal_name,
        primary_signal_type=args.primary_signal_type,
        campaign_max_age_days=args.campaign_max_age_days,
        clean_live_days=args.clean_live_days,
        check_broker=args.check_broker,
    )


def paper_soak_report_request(args: argparse.Namespace) -> PaperSoakReportRequest:
    return PaperSoakReportRequest(
        profile=args.profile,
        strategy_run_id=args.strategy_run_id,
        as_of=args.as_of,
        contracts_file=args.contracts_file,
        signal_name=args.signal_name,
        signal_type=args.signal_type,
        bar_seconds=args.bar_seconds,
        data_health_window_days=args.data_health_window_days,
        order_latency_window_days=args.order_latency_window_days,
        output=args.output,
    )


def dataset_quorum_request(args: argparse.Namespace) -> DatasetQuorumRequest:
    return DatasetQuorumRequest(
        command=args.dataset_quorum_command,
        dataset_kind=args.dataset_kind,
        as_of=args.as_of,
        vendor_bars=getattr(args, "vendor_bars", None),
        required_vendor_count=getattr(args, "required_vendor_count", 2),
        max_disagreement_bps=getattr(args, "max_disagreement_bps", Decimal("50")),
    )


def simulator_calibration_request(args: argparse.Namespace) -> SimulatorCalibrationRequest:
    return SimulatorCalibrationRequest(
        as_of=args.as_of,
        lookback_days=args.lookback_days,
        floor_bps=args.floor_bps,
        min_sample_count=args.min_sample_count,
        p90_safety_margin=args.p90_safety_margin,
        output=args.output,
    )


__all__ = [
    "dataset_quorum_request",
    "paper_soak_report_request",
    "performance_heartbeat_request",
    "performance_report_request",
    "performance_snapshot_request",
    "production_candidate_request",
    "readiness_request",
    "signal_gate_request",
    "simulator_calibration_request",
    "text_gate_request",
]
