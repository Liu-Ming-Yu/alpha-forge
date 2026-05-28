"""JSON payload helpers for production-candidate reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from quant_platform.services.governance_service.support.prediction_evidence import (
    forecast_evidence_payload,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.production import ProductionCandidateReport, SignalGateStatus


def production_candidate_payload(report: ProductionCandidateReport) -> dict[str, object]:
    """Return JSON-safe production-candidate payload."""
    representative_signal_gate = (
        _signal_gate_payload(report.representative_signal_gate_status)
        if report.representative_signal_gate_status is not None
        else None
    )
    return {
        "profile": report.profile.value,
        "generated_at": report.generated_at.isoformat(),
        "state": report.state.value,
        "passed": report.passed,
        "next_allowed_mode": report.next_allowed_mode.value,
        "promotion_blockers": list(report.promotion_blockers),
        "campaign_manifest_path": report.campaign_manifest_path,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "severity": check.severity,
            }
            for check in report.checks
        ],
        "campaign_manifest": (
            dict(report.campaign_manifest) if report.campaign_manifest is not None else None
        ),
        "representative_signal_gate_status": representative_signal_gate,
        "representative_signal_gate": representative_signal_gate,
        "signal_gates": [_signal_gate_payload(status) for status in report.signal_gate_statuses],
        "forecast_evidence": [
            forecast_evidence_payload(evidence) for evidence in report.forecast_evidence
        ],
    }


def render_production_candidate_diagnostics(payload: Mapping[str, object]) -> str:
    """Render an operator-readable diagnosis from an existing candidate payload."""
    lines = [
        "Production Candidate Diagnose",
        f"Profile: {_text(payload.get('profile'))}",
        f"State: {_text(payload.get('state'))}",
        f"Passed: {_text(payload.get('passed'))}",
        f"Next allowed mode: {_text(payload.get('next_allowed_mode'))}",
    ]
    blockers = _sequence(payload.get("promotion_blockers"))
    lines.append("Top blockers: " + (", ".join(blockers[:8]) if blockers else "none"))
    lines.extend(_campaign_lines(payload.get("campaign_manifest")))
    lines.extend(_signal_gate_lines(payload.get("signal_gates")))
    lines.extend(_forecast_evidence_lines(payload.get("forecast_evidence")))
    lines.extend(_failed_check_lines(payload.get("checks")))
    return "\n".join(lines)


def _campaign_lines(raw: object) -> list[str]:
    campaign = raw if isinstance(raw, Mapping) else {}
    if not campaign:
        return ["Campaign evidence: missing"]
    return [
        "Campaign evidence: "
        f"passed={_text(campaign.get('passed'))} "
        f"run_id={_text(campaign.get('run_id'))} "
        f"model={_text(campaign.get('model_version'))} "
        f"feature_set={_text(campaign.get('feature_set_version'))}",
    ]


def _signal_gate_lines(raw: object) -> list[str]:
    rows = [row for row in _objects(raw) if isinstance(row, Mapping)]
    if not rows:
        return ["Signal gates: none"]
    lines = ["Signal gates:"]
    for row in rows:
        lines.append(
            "  "
            f"{_text(row.get('signal_type'))}/{_text(row.get('signal_name'))}: "
            f"passed={_text(row.get('passed'))} "
            f"state={_text(row.get('state'))} "
            f"obs={_text(row.get('observations'))} "
            f"ic={_text(row.get('rolling_ic'))}"
        )
    return lines


def _forecast_evidence_lines(raw: object) -> list[str]:
    rows = [row for row in _objects(raw) if isinstance(row, Mapping)]
    if not rows:
        return ["Prediction evidence: none"]
    lines = ["Prediction evidence:"]
    for row in rows:
        lines.append(
            "  "
            f"{_text(row.get('source'))}: "
            f"passed={_text(row.get('passed'))} "
            f"horizon={_text(row.get('horizon'))} "
            f"obs={_text(row.get('observations'))} "
            f"latest={_text(row.get('latest_prediction_at'))}"
        )
    return lines


def _failed_check_lines(raw: object) -> list[str]:
    failed = [
        row for row in _objects(raw) if isinstance(row, Mapping) and row.get("passed") is not True
    ]
    if not failed:
        return ["Failed checks: none"]
    lines = ["Failed checks:"]
    for row in failed[:12]:
        lines.append(f"  {_text(row.get('name'))}: {_text(row.get('detail'))}")
    if len(failed) > 12:
        lines.append(f"  ... {len(failed) - 12} more")
    return lines


def _objects(raw: object) -> Sequence[object]:
    return raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()


def _sequence(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    return []


def _text(value: object) -> str:
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# Internal helpers


def _signal_gate_payload(status: SignalGateStatus) -> dict[str, object]:
    return {
        "signal_name": status.signal_name,
        "signal_type": status.signal_type,
        "as_of": status.as_of.isoformat(),
        "passed": status.passed,
        "state": status.state.value,
        "rolling_ic": status.rolling_ic,
        "observations": status.observations,
        "negative_streak": status.negative_streak,
        "max_drawdown": status.max_drawdown,
        "max_turnover": status.max_turnover,
    }
