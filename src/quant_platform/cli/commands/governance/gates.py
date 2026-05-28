"""Governance gate command registrations."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_platform.application.governance import (
    PerformanceHeartbeatRequest,
    PerformanceReportRequest,
    PerformanceSnapshotRequest,
    ProductionCandidateRequest,
    ReadinessRequest,
    SignalGateRequest,
    TextGateRequest,
)
from quant_platform.cli.commands.governance.request_factories import (
    performance_heartbeat_request,
    performance_report_request,
    performance_snapshot_request,
    production_candidate_request,
    readiness_request,
    signal_gate_request,
    text_gate_request,
)
from quant_platform.cli.registry import bind_command


def register_gate_commands(sub: Any) -> None:
    _register_performance(sub)
    _register_signal_gate(sub)
    _register_text_gate(sub)
    _register_readiness(sub)
    _register_production_candidate(sub)


def _register_performance(sub: Any) -> None:
    perf_p = sub.add_parser(
        "performance",
        help="Persist and report live/paper performance state.",
    )
    perf_sub = perf_p.add_subparsers(dest="performance_command", required=True)
    snap_p = perf_sub.add_parser("snapshot", help="Persist one NAV snapshot.")
    snap_p.add_argument("--strategy-run-id", required=True, type=uuid.UUID)
    snap_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    snap_p.add_argument("--nav", required=True, type=Decimal)
    snap_p.add_argument("--gross-exposure", type=Decimal, default=Decimal("0"))
    snap_p.add_argument("--cash", type=Decimal, default=Decimal("0"))
    snap_p.add_argument("--source", default="operator")
    bind_command(
        snap_p,
        use_case_name="governance.performance",
        request_factory=performance_snapshot_request,
        request_type=PerformanceSnapshotRequest,
    )

    report_p = perf_sub.add_parser("report", help="Render rolling performance report.")
    report_p.add_argument("--strategy-run-id", required=True, type=uuid.UUID)
    report_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    report_p.add_argument("--window", type=int, default=90)
    bind_command(
        report_p,
        use_case_name="governance.performance",
        request_factory=performance_report_request,
        request_type=PerformanceReportRequest,
    )

    hb_p = perf_sub.add_parser("heartbeat", help="Persist one runtime heartbeat.")
    hb_p.add_argument("--component", default="supervisor")
    hb_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    hb_p.add_argument("--status", choices=["ok", "degraded", "halted"], default="ok")
    hb_p.add_argument("--detail", default="")
    bind_command(
        hb_p,
        use_case_name="governance.performance",
        request_factory=performance_heartbeat_request,
        request_type=PerformanceHeartbeatRequest,
    )


def _register_signal_gate(sub: Any) -> None:
    sg_p = sub.add_parser(
        "signal-gate",
        help="Inspect or enforce the generic signal promotion gate.",
    )
    sg_sub = sg_p.add_subparsers(dest="signal_gate_command", required=True)
    for name in ("status", "assert"):
        gate_p = sg_sub.add_parser(name)
        gate_p.add_argument("--signal-name", required=True)
        gate_p.add_argument(
            "--signal-type",
            choices=["classical", "text", "event", "intraday", "xgboost"],
            required=True,
        )
        gate_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
        _bind_signal_gate(gate_p)
    sg_rec_p = sg_sub.add_parser("record")
    sg_rec_p.add_argument("--signal-name", required=True)
    sg_rec_p.add_argument(
        "--signal-type",
        choices=["classical", "text", "event", "intraday", "xgboost"],
        required=True,
    )
    sg_rec_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    sg_rec_p.add_argument("--daily-ic", required=True, type=float)
    sg_rec_p.add_argument("--observations", type=int, default=1)
    sg_rec_p.add_argument("--drawdown", type=float, default=0.0)
    sg_rec_p.add_argument("--turnover", type=float, default=0.0)
    _bind_signal_gate(sg_rec_p)


def _register_text_gate(sub: Any) -> None:
    tg_p = sub.add_parser(
        "text-gate",
        help="Inspect or enforce the persisted text-signal promotion gate.",
    )
    tg_sub = tg_p.add_subparsers(dest="text_gate_command", required=True)
    for name in ("status", "assert"):
        gate_p = tg_sub.add_parser(name)
        gate_p.add_argument("--strategy-name", required=True)
        gate_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
        _bind_text_gate(gate_p)
    rec_p = tg_sub.add_parser("record")
    rec_p.add_argument("--strategy-name", required=True)
    rec_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    rec_p.add_argument("--daily-ic", required=True, type=float)
    rec_p.add_argument("--observations", type=int, default=1)
    _bind_text_gate(rec_p)


def _register_readiness(sub: Any) -> None:
    ready_p = sub.add_parser(
        "readiness",
        help="Build or enforce the industrial paper/live readiness report.",
    )
    ready_sub = ready_p.add_subparsers(dest="readiness_command", required=True)
    for name in ("report", "assert"):
        r_p = ready_sub.add_parser(name)
        r_p.add_argument("--profile", choices=["paper", "live"], required=True)
        r_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
        r_p.add_argument("--contracts-file", default=None)
        r_p.add_argument("--component", default="supervisor")
        r_p.add_argument("--soak-report", type=Path, default=None)
        r_p.add_argument("--backup-manifest", type=Path, default=None)
        r_p.add_argument("--signal-name", default="")
        r_p.add_argument(
            "--signal-type",
            choices=["classical", "text", "event", "intraday", "xgboost"],
            default="classical",
        )
        r_p.add_argument(
            "--check-broker",
            action="store_true",
            help="Run the read-only broker health probe and persist the result.",
        )
        bind_command(
            r_p,
            use_case_name="governance.readiness",
            request_factory=readiness_request,
            request_type=ReadinessRequest,
        )


def _register_production_candidate(sub: Any) -> None:
    pc_p = sub.add_parser(
        "production-candidate",
        help=(
            "Aggregated promotion gate: readiness + research campaign + signal "
            "gates + V2 orchestration evidence."
        ),
    )
    pc_sub = pc_p.add_subparsers(dest="production_candidate_command", required=True)
    for name in ("diagnose", "report", "assert"):
        p = pc_sub.add_parser(name)
        p.add_argument(
            "--profile",
            choices=["paper", "llm_live_rehearsal", "live"],
            required=True,
        )
        p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
        p.add_argument("--contracts-file", default=None)
        p.add_argument("--component", default="supervisor")
        p.add_argument("--soak-report", type=Path, default=None)
        p.add_argument("--backup-manifest", type=Path, default=None)
        p.add_argument(
            "--signal-source",
            action="append",
            default=None,
            help="Repeatable.  Promoted ensemble source to enforce signal-gate "
            "evidence for.  Defaults to every source with weight > 0.",
        )
        p.add_argument(
            "--primary-signal-name",
            default="",
            help="Optional readiness primary signal name override. Defaults to "
            "the latest promoted campaign model for paper ensembles.",
        )
        p.add_argument(
            "--primary-signal-type",
            choices=["", "classical", "text", "event", "intraday", "xgboost"],
            default="",
            help="Optional readiness primary signal type override.",
        )
        p.add_argument(
            "--campaign-max-age-days",
            type=int,
            default=None,
            help="Maximum age (calendar days) of the latest research campaign "
            "manifest.  Defaults to QP__PRODUCTION__DATA_HEALTH_STALE_AFTER_DAYS.",
        )
        p.add_argument(
            "--clean-live-days",
            type=int,
            default=0,
            help="Consecutive clean live trading days from the operator runbook; "
            "controls which live ramp rung is permitted.",
        )
        p.add_argument(
            "--check-broker",
            action="store_true",
            help="Run the read-only broker health probe and persist the result.",
        )
        bind_command(
            p,
            use_case_name="governance.production_candidate",
            request_factory=production_candidate_request,
            request_type=ProductionCandidateRequest,
        )


def _bind_signal_gate(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="governance.signal_gate",
        request_factory=signal_gate_request,
        request_type=SignalGateRequest,
    )


def _bind_text_gate(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="governance.text_gate",
        request_factory=text_gate_request,
        request_type=TextGateRequest,
    )


performance = "governance.performance"
signal_gate = "governance.signal_gate"
text_gate = "governance.text_gate"
readiness = "governance.readiness"
production_candidate = "governance.production_candidate"
