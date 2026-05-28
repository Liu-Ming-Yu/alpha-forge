"""Operational evidence checks for production readiness reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    BrokerSmokeObservation,
    PaperLifecycleObservation,
    PreflightCheck,
    ProductionProfile,
    SignalGateStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import OperationalReadinessRepository


async def repository_evidence_checks(
    repo: OperationalReadinessRepository,
    *,
    settings: PlatformSettings,
    profile: ProductionProfile,
    as_of: datetime,
    component: str,
    broker_checked: bool,
    current_broker_smoke: BrokerSmokeObservation | None,
    current_paper_lifecycle: PaperLifecycleObservation | None,
) -> list[PreflightCheck]:
    """Build checks from persisted heartbeat, broker, smoke, and lifecycle evidence."""
    live = profile == ProductionProfile.LIVE
    checks: list[PreflightCheck] = []

    heartbeat = await repo.latest_runtime_heartbeat(component)
    heartbeat_fresh = False
    if heartbeat is not None:
        stale_after = timedelta(minutes=settings.production.heartbeat_stale_after_minutes)
        heartbeat_fresh = heartbeat.status == "ok" and heartbeat.as_of >= as_of - stale_after
    checks.append(
        PreflightCheck(
            name="runtime_heartbeat_fresh",
            passed=heartbeat_fresh,
            detail=(
                f"{component} heartbeat at {heartbeat.as_of.isoformat()}"
                if heartbeat
                else f"no heartbeat for {component}"
            ),
        )
    )

    latest_broker = await repo.latest_broker_health()
    broker_ready = latest_broker is not None and latest_broker.status == "connected"
    latest_smoke = current_broker_smoke or await repo.latest_broker_smoke()
    smoke_ready = latest_smoke is not None and latest_smoke.passed
    latest_lifecycle = current_paper_lifecycle or await repo.latest_paper_lifecycle()
    lifecycle_ready = latest_lifecycle is not None and latest_lifecycle.passed

    checks.append(
        PreflightCheck(
            name="broker_check_requested",
            passed=broker_checked,
            detail="--check-broker was supplied"
            if broker_checked
            else "--check-broker is required",
            severity="error" if live else "warning",
        )
    )
    checks.append(
        PreflightCheck(
            name="broker_health_persisted",
            passed=broker_ready,
            detail=(
                f"{latest_broker.status} at {latest_broker.observed_at.isoformat()}"
                if latest_broker
                else "no broker health observation"
            ),
            severity="error" if broker_checked else "warning",
        )
    )
    checks.append(
        PreflightCheck(
            name="broker_smoke_persisted",
            passed=smoke_ready,
            detail=(
                f"{latest_smoke.status} {latest_smoke.host}:{latest_smoke.port} "
                f"client_id={latest_smoke.client_id} at {latest_smoke.observed_at.isoformat()}"
                if latest_smoke
                else "no broker smoke observation"
            ),
            severity="error" if live or broker_checked else "warning",
        )
    )
    checks.append(
        PreflightCheck(
            name="paper_lifecycle_persisted",
            passed=lifecycle_ready,
            detail=(
                f"{latest_lifecycle.status} broker_order_id={latest_lifecycle.broker_order_id} "
                f"at {latest_lifecycle.observed_at.isoformat()}"
                if latest_lifecycle
                else "no paper lifecycle observation"
            ),
            severity="error" if live else "warning",
        )
    )
    return checks


def signal_gate_check(
    signal_status: SignalGateStatus | None,
    *,
    live: bool,
) -> PreflightCheck:
    return PreflightCheck(
        name="signal_gate_passed",
        passed=signal_status is not None and signal_status.passed,
        detail=(
            f"{signal_status.signal_type}/{signal_status.signal_name} "
            f"state={signal_status.state.value} rolling_ic={signal_status.rolling_ic:.4f} "
            f"drawdown={signal_status.max_drawdown:.4f} turnover={signal_status.max_turnover:.4f}"
            if signal_status is not None
            else "signal gate evidence is required"
        ),
        severity="error" if live or signal_status is not None else "warning",
    )


def alpha_policy_checks(
    settings: PlatformSettings, *, profile: ProductionProfile
) -> list[PreflightCheck]:
    live = profile == ProductionProfile.LIVE
    return [
        PreflightCheck(
            name="alpha_ensemble_mode_governed",
            passed=settings.alpha.ensemble_mode in {"shadow", "paper", "live"},
            detail=(
                f"ensemble_mode={settings.alpha.ensemble_mode} "
                f"paper_max_non_classical_weight={settings.alpha.paper_max_non_classical_weight} "
                f"live_max_non_classical_weight={settings.alpha.max_non_classical_weight}"
            ),
        ),
        PreflightCheck(
            name="alpha_live_cap_conservative",
            passed=(
                not live
                or settings.alpha.max_non_classical_weight
                <= float(settings.alpha.live_ramp_initial)
            ),
            detail=(
                f"max_non_classical_weight={settings.alpha.max_non_classical_weight} "
                f"initial_ramp={settings.alpha.live_ramp_initial}"
            ),
        ),
        PreflightCheck(
            name="alpha_fail_closed_enabled",
            passed=settings.alpha.fail_closed_on_promoted_source_error,
            detail=(
                "promoted source failures block the cycle"
                if settings.alpha.fail_closed_on_promoted_source_error
                else "promoted source failures may be ignored"
            ),
            severity="error" if live else "warning",
        ),
    ]


def backup_manifest_check(path: Path | None, *, live: bool) -> PreflightCheck:
    return PreflightCheck(
        name="backup_restore_manifest_present",
        passed=path is not None and path.is_file(),
        detail=str(path) if path is not None else "backup/restore manifest is required",
        severity="error" if live else "warning",
    )
