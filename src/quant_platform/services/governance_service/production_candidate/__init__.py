"""Aggregated production-candidate gate.

Composes readiness, research-campaign, and signal-gate evidence into one
operator-facing decision: which operating mode is currently allowed.

The detailed checks live in focused helper modules so this file remains the
read-only orchestration facade for the production-candidate use case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    BrokerSmokeObservation,
    ForecastEvidence,
    PaperLifecycleObservation,
    PreflightCheck,
    ProductionCandidateReport,
    ProductionProfile,
    ProductionReadinessReport,
    SignalGateStatus,
)
from quant_platform.services.governance_service.gates.signal_gate import signal_gate_status
from quant_platform.services.governance_service.llm_live_startup import (
    build_llm_live_evidence_checks,
)
from quant_platform.services.governance_service.production_candidate.campaign import (
    _backtest_evidence_manifest_check,
    _campaign_manifest_check,
    _campaign_passed_check,
    _campaign_weights_cap_check,
    _migration_head_check,
)
from quant_platform.services.governance_service.production_candidate.llm import (
    llm_live_rehearsal_config_checks,
    llm_shadow_paper_parity_check,
)
from quant_platform.services.governance_service.production_candidate.payload import (
    production_candidate_payload,
)
from quant_platform.services.governance_service.production_candidate.promotion import (
    _compute_promotion_mode,
)
from quant_platform.services.governance_service.production_candidate.signal import (
    _prediction_evidence_check,
    _signal_gate_check,
)
from quant_platform.services.governance_service.production_candidate.sources import (
    CLASSICAL_SOURCE,
    _primary_signal_status,
    _resolve_signal_sources,
)
from quant_platform.services.governance_service.production_candidate.v2 import (
    _v2_dataset_quorum_evidence_check,
    _v2_live_path_checks,
)
from quant_platform.services.governance_service.readiness import (
    _readiness_state,
    build_readiness_report,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import DatasetCatalog, GovernanceEvidenceRepository
    from quant_platform.services.governance_service.preflight import (
        BrokerHealthChecker,
        DataFreshnessProvider,
    )

__all__ = ["build_production_candidate_report", "production_candidate_payload"]


def build_performance_repository(_dsn: str | None) -> GovernanceEvidenceRepository:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("production-candidate repository must be supplied by bootstrap")


async def build_production_candidate_report(
    settings: PlatformSettings,
    *,
    profile: ProductionProfile,
    as_of: datetime,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None = None,
    backup_manifest: Path | None = None,
    component: str = "supervisor",
    broker_checked: bool = False,
    current_broker_smoke: BrokerSmokeObservation | None = None,
    current_paper_lifecycle: PaperLifecycleObservation | None = None,
    broker_gateway: BrokerHealthChecker | None = None,
    data_provider: DataFreshnessProvider | None = None,
    reference_instrument_id: uuid.UUID | None = None,
    signal_sources: Iterable[str] | None = None,
    primary_signal_name: str | None = None,
    primary_signal_type: str | None = None,
    campaign_max_age_days: int | None = None,
    clean_live_days: int = 0,
    evidence_repository: GovernanceEvidenceRepository | None = None,
    dataset_catalog: DatasetCatalog | None = None,
    packaged_migration_head: str | None = None,
) -> ProductionCandidateReport:
    """Evaluate the aggregated production-candidate gate."""
    sources = _resolve_signal_sources(signal_sources, settings)
    evidence_repo = evidence_repository or build_performance_repository(
        settings.storage.postgres_dsn
    )

    extra_checks: list[PreflightCheck] = []
    extra_checks.append(_migration_head_check(settings, packaged_head=packaged_migration_head))

    _campaign_path, campaign_payload, campaign_check = _campaign_manifest_check(
        settings,
        as_of=as_of,
        max_age_days=campaign_max_age_days,
    )
    primary_signal_status = await _primary_signal_status(
        settings,
        sources,
        as_of,
        campaign_payload=campaign_payload,
        signal_sources_explicit=signal_sources is not None,
        signal_name=primary_signal_name,
        signal_type=primary_signal_type,
        signal_gate=evidence_repo,
    )

    readiness = await build_readiness_report(
        settings,
        profile=profile,
        as_of=as_of,
        instrument_contracts=instrument_contracts,
        soak_report=soak_report,
        backup_manifest=backup_manifest,
        signal_status=primary_signal_status,
        current_broker_smoke=current_broker_smoke,
        current_paper_lifecycle=current_paper_lifecycle,
        component=component,
        broker_checked=broker_checked,
        broker_gateway=broker_gateway,
        data_provider=data_provider,
        reference_instrument_id=reference_instrument_id,
        evidence_repository=evidence_repo,
    )

    extra_checks.append(campaign_check)

    if campaign_payload is not None:
        extra_checks.append(_campaign_passed_check(campaign_payload))
        extra_checks.append(_campaign_weights_cap_check(campaign_payload, settings))
        extra_checks.append(_backtest_evidence_manifest_check(campaign_payload, settings))

    signal_statuses: list[SignalGateStatus] = []
    forecast_evidence_rows: list[ForecastEvidence] = []
    text_source_checked = False
    for source in sources:
        if source == CLASSICAL_SOURCE:
            continue
        status = await signal_gate_status(
            settings,
            signal_name=source,
            signal_type=source,
            as_of=as_of,
            gate=evidence_repo,
        )
        signal_statuses.append(status)
        extra_checks.append(_signal_gate_check(source, status, profile))
        evidence = await evidence_repo.forecast_evidence(
            source,
            as_of=as_of,
            stale_after_hours=settings.production.prediction_evidence_stale_after_hours,
            min_confidence=settings.production.prediction_evidence_min_confidence,
        )
        forecast_evidence_rows.append(evidence)
        extra_checks.append(_prediction_evidence_check(source, evidence, profile))
        if source == "text" and settings.llm.live_mode_enabled:
            text_source_checked = True
            extra_checks.extend(
                build_llm_live_evidence_checks(
                    settings,
                    as_of=as_of,
                    forecast_evidence=evidence,
                    profile=profile,
                )
            )

    if profile == ProductionProfile.LLM_LIVE_REHEARSAL:
        extra_checks.extend(llm_live_rehearsal_config_checks(settings, sources))

    if settings.llm.live_mode_enabled and profile in {
        ProductionProfile.LIVE,
        ProductionProfile.LLM_LIVE_REHEARSAL,
    }:
        if not text_source_checked:
            extra_checks.append(
                PreflightCheck(
                    name="llm_live_text_source_positive_weight",
                    passed=False,
                    detail="live LLM mode requires text in promoted signal sources",
                )
            )
        extra_checks.append(
            await llm_shadow_paper_parity_check(
                evidence_repo,
                as_of=as_of,
            )
        )

    extra_checks.extend(_v2_live_path_checks(settings, profile))
    if settings.v2.require_dataset_quorum:
        if dataset_catalog is not None:
            quorum_check = await _v2_dataset_quorum_evidence_check(
                settings,
                as_of=as_of,
                profile=profile,
                dataset_catalog=dataset_catalog,
            )
        else:
            quorum_check = await _v2_dataset_quorum_evidence_check(
                settings,
                as_of=as_of,
                profile=profile,
            )
        extra_checks.append(quorum_check)

    combined_checks = tuple(readiness.checks) + tuple(extra_checks)
    state = _readiness_state(list(combined_checks))

    next_allowed, blockers = _compute_promotion_mode(
        profile=profile,
        state=state,
        checks=combined_checks,
        clean_live_days=clean_live_days,
        settings=settings,
    )

    aggregated = ProductionReadinessReport(
        profile=profile,
        generated_at=readiness.generated_at,
        state=state,
        checks=combined_checks,
    )

    return ProductionCandidateReport(
        profile=profile,
        generated_at=readiness.generated_at,
        state=state,
        next_allowed_mode=next_allowed,
        promotion_blockers=blockers,
        checks=combined_checks,
        readiness=aggregated,
        campaign_manifest_path=str(_campaign_path) if _campaign_path is not None else None,
        campaign_manifest=campaign_payload,
        representative_signal_gate_status=primary_signal_status,
        signal_gate_statuses=tuple(signal_statuses),
        forecast_evidence=tuple(forecast_evidence_rows),
    )
