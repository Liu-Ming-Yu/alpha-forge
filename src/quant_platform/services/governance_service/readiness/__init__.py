"""Industrial deployment-readiness checks."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    BrokerSmokeObservation,
    PaperLifecycleObservation,
    PreflightCheck,
    ProductionProfile,
    ProductionReadinessReport,
    ReadinessSnapshot,
    SignalGateStatus,
)
from quant_platform.services.governance_service.preflight import (
    BrokerHealthChecker,
    DataFreshnessProvider,
    check_broker_connectivity,
    check_data_freshness,
    evaluate_preflight,
)
from quant_platform.services.governance_service.readiness.readiness_checks import (
    paper_soak_check,
    readiness_state,
)
from quant_platform.services.governance_service.readiness.readiness_evidence_checks import (
    alpha_policy_checks,
    backup_manifest_check,
    repository_evidence_checks,
    signal_gate_check,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import OperationalReadinessRepository

_readiness_state = readiness_state


def build_performance_repository(_dsn: str | None) -> OperationalReadinessRepository:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("readiness repository must be supplied by bootstrap")


async def build_readiness_report(
    settings: PlatformSettings,
    *,
    profile: ProductionProfile,
    as_of: datetime,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None = None,
    backup_manifest: Path | None = None,
    signal_status: SignalGateStatus | None = None,
    current_broker_smoke: BrokerSmokeObservation | None = None,
    current_paper_lifecycle: PaperLifecycleObservation | None = None,
    component: str = "supervisor",
    broker_checked: bool = False,
    broker_gateway: BrokerHealthChecker | None = None,
    data_provider: DataFreshnessProvider | None = None,
    reference_instrument_id: uuid.UUID | None = None,
    evidence_repository: OperationalReadinessRepository | None = None,
) -> ProductionReadinessReport:
    """Build a promotion gate report for paper-soak or live-readiness workflows."""
    checks: list[PreflightCheck] = list(
        evaluate_preflight(
            settings,
            profile=profile,
            instrument_contracts=instrument_contracts,
        ).checks
    )
    repo = evidence_repository or build_performance_repository(settings.storage.postgres_dsn)

    live = profile == ProductionProfile.LIVE
    checks.extend(
        await repository_evidence_checks(
            repo,
            settings=settings,
            profile=profile,
            as_of=as_of,
            component=component,
            broker_checked=broker_checked,
            current_broker_smoke=current_broker_smoke,
            current_paper_lifecycle=current_paper_lifecycle,
        )
    )

    checks.append(signal_gate_check(signal_status, live=live))
    checks.extend(alpha_policy_checks(settings, profile=profile))

    # Live runtime checks — only run when the relevant objects are supplied.
    if broker_gateway is not None:
        checks.append(await check_broker_connectivity(broker_gateway))
    if data_provider is not None and reference_instrument_id is not None:
        checks.append(
            await check_data_freshness(
                data_provider,
                reference_instrument_id,
                bar_seconds=86400,
                max_age_minutes=settings.risk.max_data_age_minutes,
            )
        )

    checks.append(
        paper_soak_check(
            soak_report,
            as_of=as_of,
            stale_after_days=settings.production.data_health_stale_after_days,
            live=live,
        )
    )
    checks.append(backup_manifest_check(backup_manifest, live=live))

    state = readiness_state(checks)
    return ProductionReadinessReport(
        profile=profile,
        generated_at=as_of.astimezone(UTC),
        state=state,
        checks=tuple(checks),
    )


def readiness_payload(report: ProductionReadinessReport) -> dict[str, object]:
    """Return JSON-safe readiness report payload."""
    return {
        "profile": report.profile.value,
        "generated_at": report.generated_at.isoformat(),
        "state": report.state.value,
        "passed": report.passed,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "severity": check.severity,
            }
            for check in report.checks
        ],
    }


async def persist_readiness_snapshot(
    report: ProductionReadinessReport,
    repository: object,
) -> ReadinessSnapshot:
    """Persist a V2 readiness snapshot using a ProductionEvidenceRepository."""
    snapshot = ReadinessSnapshot(
        snapshot_id=uuid.uuid4(),
        profile=report.profile,
        generated_at=report.generated_at,
        state=report.state,
        passed=report.passed,
        checks=report.checks,
    )
    save = getattr(repository, "save_readiness_snapshot", None)
    if not callable(save):
        raise TypeError("repository must provide save_readiness_snapshot")
    await save(snapshot)
    return snapshot
