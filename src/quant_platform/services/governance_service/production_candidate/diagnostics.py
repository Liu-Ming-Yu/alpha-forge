"""Human-readable diagnostics for production-candidate reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.core.domain.production import (
        ForecastEvidence,
        PreflightCheck,
        ProductionCandidateReport,
        SignalGateStatus,
    )


_CAMPAIGN_CHECK_NAMES = (
    "research_campaign_manifest_present",
    "research_campaign_eligibility_passed",
    "research_campaign_paper_weights_within_cap",
    "intraday_backtest_evidence_manifest_passed",
)


def render_production_candidate_diagnostics(report: ProductionCandidateReport) -> str:
    """Render an operator-facing evidence inventory.

    This intentionally mirrors the production-candidate report instead of
    re-evaluating gates, so ``diagnose`` cannot drift away from ``report`` or
    ``assert`` semantics.
    """
    lines: list[str] = []
    lines.extend(_status_lines(report))
    lines.append("")
    lines.extend(_campaign_lines(report))
    lines.append("")
    lines.extend(_signal_gate_lines(report))
    lines.append("")
    lines.extend(_forecast_evidence_lines(report.forecast_evidence))
    lines.append("")
    lines.extend(_blocker_lines(report))
    return "\n".join(lines)


def _status_lines(report: ProductionCandidateReport) -> list[str]:
    return [
        "Production candidate status",
        f"profile={report.profile.value}",
        f"readiness_state={report.state.value}",
        f"next_allowed_mode={report.next_allowed_mode.value}",
        f"passed={_bool(report.passed)}",
    ]


def _campaign_lines(report: ProductionCandidateReport) -> list[str]:
    checks = _check_map(report.checks)
    lines = [
        "Campaign evidence",
        f"campaign_manifest_path={report.campaign_manifest_path or 'none'}",
    ]
    for name in _CAMPAIGN_CHECK_NAMES:
        check = checks.get(name)
        if check is None:
            continue
        lines.append(_check_line(check))
    return lines


def _signal_gate_lines(report: ProductionCandidateReport) -> list[str]:
    lines = ["Signal gates"]
    statuses = _dedupe_signal_statuses(
        (report.representative_signal_gate_status, *report.signal_gate_statuses)
    )
    if not statuses:
        lines.append("none")
        return lines
    for status in statuses:
        lines.append(
            f"{status.signal_name}: observations={status.observations} "
            f"rolling_ic={status.rolling_ic:.4f} "
            f"negative_streak={status.negative_streak} "
            f"drawdown={status.max_drawdown:.4f} "
            f"turnover={status.max_turnover:.4f} "
            f"passed={_bool(status.passed)}"
        )
    return lines


def _forecast_evidence_lines(evidence_rows: Iterable[ForecastEvidence]) -> list[str]:
    lines = ["Prediction evidence"]
    rows = list(evidence_rows)
    if not rows:
        lines.append("none")
        return lines
    for evidence in rows:
        latest = (
            evidence.latest_prediction_at.isoformat()
            if evidence.latest_prediction_at is not None
            else "none"
        )
        blockers = "; ".join(evidence.blockers) if evidence.blockers else "none"
        lines.append(
            f"{evidence.source}: source={evidence.source} "
            f"observations={evidence.observations} "
            f"latest_prediction_at={latest} "
            f"stale={_bool(evidence.stale)} "
            f"mean_confidence={evidence.mean_confidence:.4f} "
            f"passed={_bool(evidence.passed)} "
            f"blockers={blockers}"
        )
    return lines


def _blocker_lines(report: ProductionCandidateReport) -> list[str]:
    lines = ["Top blockers"]
    checks = _check_map(report.checks)
    blockers = list(report.promotion_blockers)
    if not blockers:
        lines.append("none")
        return lines
    for idx, name in enumerate(blockers, start=1):
        check = checks.get(name)
        detail = check.detail if check is not None else ""
        suffix = f" - {detail}" if detail else ""
        lines.append(f"{idx}. {name}{suffix}")
    return lines


def _check_map(checks: Iterable[PreflightCheck]) -> dict[str, PreflightCheck]:
    return {check.name: check for check in checks}


def _check_line(check: PreflightCheck) -> str:
    return (
        f"{check.name}: passed={_bool(check.passed)} "
        f"severity={check.severity} detail={check.detail}"
    )


def _dedupe_signal_statuses(
    statuses: Iterable[SignalGateStatus | None],
) -> list[SignalGateStatus]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SignalGateStatus] = []
    for status in statuses:
        if status is None:
            continue
        key = (status.signal_name, status.signal_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(status)
    return deduped


def _bool(value: bool) -> str:
    return "true" if value else "false"
