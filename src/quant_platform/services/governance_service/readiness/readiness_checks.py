"""Pure readiness check helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    PreflightCheck,
    ReadinessState,
)

if TYPE_CHECKING:
    from pathlib import Path


def paper_soak_check(
    path: Path | None,
    *,
    as_of: datetime,
    stale_after_days: int,
    live: bool,
) -> PreflightCheck:
    severity = "error" if live else "warning"
    if path is None:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail="soak report is required",
            severity=severity,
        )
    if not path.is_file():
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} does not exist",
            severity=severity,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} is not valid JSON: {exc}",
            severity=severity,
        )
    if not isinstance(payload, dict):
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} must contain a JSON object",
            severity=severity,
        )

    required = {
        "generated_at",
        "broker_health",
        "lifecycle_result",
        "nav_snapshot",
        "data_health",
        "signal_gate",
        "prediction_quality",
        "reconciliation",
        "order_latency",
    }
    missing = sorted(required - set(payload))
    if missing:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} missing fields: {', '.join(missing)}",
            severity=severity,
        )

    try:
        generated_at = datetime.fromisoformat(str(payload["generated_at"]))
    except ValueError as exc:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} generated_at is invalid: {exc}",
            severity=severity,
        )
    if generated_at.tzinfo is None:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} generated_at must be timezone-aware",
            severity=severity,
        )
    stale_after = timedelta(days=max(1, stale_after_days))
    if generated_at < as_of - stale_after:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} generated_at {generated_at.isoformat()} is stale",
            severity=severity,
        )

    bad_sections: list[str] = []
    for section in ("broker_health", "lifecycle_result", "data_health", "signal_gate"):
        value = payload.get(section)
        if not isinstance(value, dict) or value.get("passed") is not True:
            bad_sections.append(section)
    prediction_quality = payload.get("prediction_quality")
    if not isinstance(prediction_quality, list) or any(
        isinstance(row, dict) and row.get("passed") is False for row in prediction_quality
    ):
        bad_sections.append("prediction_quality")
    reconciliation = payload.get("reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("drift_detected") is True:
        bad_sections.append("reconciliation")
    if bad_sections:
        return PreflightCheck(
            name="paper_soak_report_valid",
            passed=False,
            detail=f"{path} failed sections: {', '.join(bad_sections)}",
            severity=severity,
        )

    return PreflightCheck(
        name="paper_soak_report_valid",
        passed=True,
        detail=str(path),
        severity=severity,
    )


def readiness_state(checks: list[PreflightCheck]) -> ReadinessState:
    errors = [check for check in checks if not check.passed and check.severity == "error"]
    if not errors:
        return ReadinessState.READY
    if any(
        check.name
        in {
            "signal_gate_passed",
            "broker_health_persisted",
            "broker_smoke_persisted",
            "paper_lifecycle_persisted",
        }
        for check in errors
    ):
        return ReadinessState.HALTED
    return ReadinessState.DEGRADED
