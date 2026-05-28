"""Governance evidence command registrations."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_platform.application.governance import (
    DatasetQuorumRequest,
    PaperSoakReportRequest,
    SimulatorCalibrationRequest,
)
from quant_platform.cli.commands.governance.request_factories import (
    dataset_quorum_request,
    paper_soak_report_request,
    simulator_calibration_request,
)
from quant_platform.cli.registry import bind_command


def register_evidence_commands(sub: Any) -> None:
    _register_paper_soak(sub)
    _register_dataset_quorum(sub)
    _register_simulator_calibration(sub)


def _register_paper_soak(sub: Any) -> None:
    ps_p = sub.add_parser(
        "paper-soak",
        help=(
            "Generate a machine-owned paper-soak evidence JSON from "
            "persisted runtime state for the readiness/production-candidate gates."
        ),
    )
    ps_sub = ps_p.add_subparsers(dest="paper_soak_command", required=True)
    rep_p = ps_sub.add_parser(
        "report",
        help="Build a paper-soak report from broker, lifecycle, NAV, signal-gate, and "
        "fill-event evidence.",
    )
    rep_p.add_argument("--profile", choices=["paper", "live"], default="paper")
    rep_p.add_argument(
        "--strategy-run-id",
        required=True,
        type=uuid.UUID,
        help="Strategy run scope for NAV evidence.",
    )
    rep_p.add_argument(
        "--as-of",
        required=True,
        type=datetime.fromisoformat,
        help="Cut-off timestamp for staleness windows and data-health.",
    )
    rep_p.add_argument(
        "--contracts-file",
        default=None,
        help="Optional contracts JSON used to materialise the data-health section.",
    )
    rep_p.add_argument("--signal-name", default="")
    rep_p.add_argument(
        "--signal-type",
        choices=["classical", "text", "event", "intraday", "xgboost"],
        default="classical",
    )
    rep_p.add_argument("--bar-seconds", type=int, default=86400)
    rep_p.add_argument("--data-health-window-days", type=int, default=5)
    rep_p.add_argument("--order-latency-window-days", type=int, default=7)
    rep_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path the soak report JSON will be written to.  Defaults to "
        "$QP__STORAGE__OBJECT_STORE_ROOT/paper_soak/<as_of>.json so the "
        "operator API and readiness gate can pick it up automatically.",
    )
    bind_command(
        rep_p,
        use_case_name="governance.paper_soak",
        request_factory=paper_soak_report_request,
        request_type=PaperSoakReportRequest,
    )


def _register_dataset_quorum(sub: Any) -> None:
    dq_p = sub.add_parser(
        "dataset-quorum",
        help=(
            "Record / inspect persisted vendor-quorum evidence consumed by "
            "the production-candidate gate (R-DAT-04 closure)."
        ),
    )
    dq_sub = dq_p.add_subparsers(dest="dataset_quorum_command", required=True)

    rec_p = dq_sub.add_parser(
        "record",
        help="Persist vendor quorum evidence built from a JSON payload of vendor closes.",
    )
    rec_p.add_argument("--dataset-kind", default="bars_eod")
    rec_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    rec_p.add_argument(
        "--vendor-bars",
        required=True,
        type=Path,
        help="JSON file mapping vendor name to a list of bar payloads "
        '(``{"instrument_id": "...", "timestamp": "...", "close": "..."}``).',
    )
    rec_p.add_argument("--required-vendor-count", type=int, default=2)
    rec_p.add_argument("--max-disagreement-bps", type=Decimal, default=Decimal("50"))
    bind_command(
        rec_p,
        use_case_name="governance.dataset_quorum",
        request_factory=dataset_quorum_request,
        request_type=DatasetQuorumRequest,
    )

    latest_p = dq_sub.add_parser(
        "latest",
        help="Print the most recent persisted quorum evidence for a dataset_kind.",
    )
    latest_p.add_argument("--dataset-kind", default="bars_eod")
    latest_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    bind_command(
        latest_p,
        use_case_name="governance.dataset_quorum",
        request_factory=dataset_quorum_request,
        request_type=DatasetQuorumRequest,
    )


def _register_simulator_calibration(sub: Any) -> None:
    sc_p = sub.add_parser(
        "simulator-calibration",
        help=(
            "Compare paper fill_events.slippage_bps against the configured "
            "ParticipationFillModel to produce a tactic-aware calibration "
            "artifact consumed by the research campaign."
        ),
    )
    sc_sub = sc_p.add_subparsers(dest="simulator_calibration_command", required=True)
    rep_p = sc_sub.add_parser(
        "report",
        help="Build a calibration report from recent paper fills.",
    )
    rep_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    rep_p.add_argument("--lookback-days", type=int, default=30)
    rep_p.add_argument("--floor-bps", type=float, default=1.0)
    rep_p.add_argument("--min-sample-count", type=int, default=20)
    rep_p.add_argument("--p90-safety-margin", type=float, default=0.0)
    rep_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path.  Defaults to "
        "$QP__STORAGE__OBJECT_STORE_ROOT/calibration/simulator_calibration_<as_of>.json.",
    )
    bind_command(
        rep_p,
        use_case_name="governance.simulator_calibration",
        request_factory=simulator_calibration_request,
        request_type=SimulatorCalibrationRequest,
    )


paper_soak = "governance.paper_soak"
dataset_quorum = "governance.dataset_quorum"
simulator_calibration = "governance.simulator_calibration"
