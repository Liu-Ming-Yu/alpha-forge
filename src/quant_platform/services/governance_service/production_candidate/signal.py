"""Signal-gate evidence checks for production-candidate reports."""

from __future__ import annotations

from quant_platform.core.domain.production import (
    ForecastEvidence,
    PreflightCheck,
    ProductionProfile,
    SignalGateStatus,
)


def _signal_gate_check(
    source: str,
    status: SignalGateStatus,
    profile: ProductionProfile,
) -> PreflightCheck:
    severity = "error" if profile == ProductionProfile.LIVE or source != "classical" else "warning"
    return PreflightCheck(
        name=f"signal_gate_{source}_passed",
        passed=status.passed,
        detail=(
            f"{source} state={status.state.value} "
            f"rolling_ic={status.rolling_ic:.4f} "
            f"observations={status.observations} "
            f"negative_streak={status.negative_streak} "
            f"max_drawdown={status.max_drawdown:.4f} "
            f"max_turnover={status.max_turnover:.4f}"
        ),
        severity=severity,
    )


def _prediction_evidence_check(
    source: str,
    evidence: ForecastEvidence,
    profile: ProductionProfile,
) -> PreflightCheck:
    severity = "error" if profile == ProductionProfile.LIVE or source != "classical" else "warning"
    details = [
        f"source={source}",
        f"observations={evidence.observations}",
        f"mean_confidence={evidence.mean_confidence:.4f}",
        f"stale={evidence.stale}",
    ]
    if evidence.latest_prediction_at is not None:
        details.append(f"latest_prediction_at={evidence.latest_prediction_at.isoformat()}")
    if evidence.blockers:
        details.append(f"blockers={'; '.join(evidence.blockers)}")
    return PreflightCheck(
        name=f"prediction_evidence_{source}_fresh",
        passed=evidence.passed,
        detail=" ".join(details),
        severity=severity,
    )
