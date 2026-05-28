"""Governance CLI operation public facade."""

from __future__ import annotations

from quant_platform.application.operator.serialization import _json_default
from quant_platform.bootstrap.governance.alpha import alpha_command
from quant_platform.bootstrap.governance.commands import (
    performance_heartbeat_command,
    performance_report_command,
    performance_snapshot_command,
    preflight_payload,
    smoke_command,
)
from quant_platform.bootstrap.governance.dataset import dataset_quorum_command
from quant_platform.bootstrap.governance.evidence import (
    paper_soak_report_command,
    simulator_calibration_command,
)
from quant_platform.bootstrap.governance.gates import signal_gate_command, text_gate_command
from quant_platform.bootstrap.governance.readiness import (
    production_candidate_diagnostics_for_cli,
    production_candidate_payload_for_cli,
    readiness_payload_for_cli,
)

__all__ = [
    "_json_default",
    "alpha_command",
    "dataset_quorum_command",
    "paper_soak_report_command",
    "performance_heartbeat_command",
    "performance_report_command",
    "performance_snapshot_command",
    "preflight_payload",
    "production_candidate_diagnostics_for_cli",
    "production_candidate_payload_for_cli",
    "readiness_payload_for_cli",
    "signal_gate_command",
    "simulator_calibration_command",
    "smoke_command",
    "text_gate_command",
]
