"""Governance and readiness command registrations."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from quant_platform.application.data import DataHealthRequest
from quant_platform.application.operator.requests import PreflightRequest
from quant_platform.cli.commands.governance.evidence import (
    dataset_quorum,
    paper_soak,
    register_evidence_commands,
    simulator_calibration,
)
from quant_platform.cli.commands.governance.gates import (
    performance,
    production_candidate,
    readiness,
    register_gate_commands,
    signal_gate,
    text_gate,
)
from quant_platform.cli.registry import bind_command


def register(sub: Any) -> None:
    pre_p = sub.add_parser(
        "preflight",
        help="Run production-readiness checks for paper or live deployment.",
    )
    pre_p.add_argument("--profile", choices=["paper", "live"], required=True)
    pre_p.add_argument("--contracts-file", default=None)
    bind_command(
        pre_p,
        use_case_name="governance.preflight",
        request_factory=lambda args: PreflightRequest(
            profile=args.profile,
            contracts_file=args.contracts_file,
        ),
        request_type=PreflightRequest,
    )

    dh_p = sub.add_parser(
        "data-health",
        help="Report bar/liquidity coverage for the configured universe.",
    )
    dh_p.add_argument("--contracts-file", required=True)
    dh_p.add_argument("--start", required=True, type=date.fromisoformat)
    dh_p.add_argument("--end", required=True, type=date.fromisoformat)
    dh_p.add_argument("--bar-seconds", type=int, default=86400)
    bind_command(
        dh_p,
        use_case_name="data.health",
        request_factory=lambda args: DataHealthRequest(
            contracts_file=Path(args.contracts_file),
            start=args.start,
            end=args.end,
            bar_seconds=args.bar_seconds,
        ),
        request_type=DataHealthRequest,
    )

    register_gate_commands(sub)
    register_evidence_commands(sub)


__all__ = [
    "dataset_quorum",
    "paper_soak",
    "performance",
    "production_candidate",
    "readiness",
    "register",
    "signal_gate",
    "simulator_calibration",
    "text_gate",
]
