"""Readiness and production-candidate governance payload wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant_platform.bootstrap.data.v2_datasets import build_dataset_catalog
from quant_platform.bootstrap.governance.repositories import build_performance_repository
from quant_platform.bootstrap.persistence.migrations import (
    alembic_packaged_head,
    verify_postgres_schema,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import (
        BrokerSmokeObservation,
        ProductionCandidateReport,
    )


async def readiness_payload_for_cli(
    settings: PlatformSettings,
    *,
    profile: str,
    as_of: datetime,
    signal_name: str,
    signal_type: str,
    backup_manifest: Path | None,
    component: str,
    check_broker: bool,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None,
    broker_smoke: BrokerSmokeObservation | None,
) -> tuple[dict[str, Any], bool]:
    from quant_platform.core.domain.production import ProductionProfile
    from quant_platform.services.governance_service.gates.signal_gate import signal_gate_status
    from quant_platform.services.governance_service.readiness import (
        build_readiness_report,
        readiness_payload,
    )

    await verify_postgres_schema(settings)
    evidence_repo = build_performance_repository(settings.storage.postgres_dsn)
    signal_status = None
    if signal_name:
        signal_status = await signal_gate_status(
            settings,
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            gate=evidence_repo,
        )
    elif settings.alpha.ensemble_mode == "paper":
        from quant_platform.services.governance_service.production_candidate.campaign import (
            _campaign_manifest_check,
        )
        from quant_platform.services.governance_service.production_candidate.sources import (
            _primary_signal_status,
            _resolve_signal_sources,
        )

        _campaign_path, campaign_payload, _campaign_check = _campaign_manifest_check(
            settings,
            as_of=as_of,
            max_age_days=None,
        )
        signal_status = await _primary_signal_status(
            settings,
            _resolve_signal_sources(None, settings),
            as_of,
            campaign_payload=campaign_payload,
            signal_gate=evidence_repo,
        )
    report = await build_readiness_report(
        settings,
        profile=ProductionProfile(profile),
        as_of=as_of,
        instrument_contracts=instrument_contracts,
        soak_report=soak_report,
        backup_manifest=backup_manifest,
        signal_status=signal_status,
        current_broker_smoke=broker_smoke,
        component=component,
        broker_checked=check_broker,
        evidence_repository=evidence_repo,
    )
    return readiness_payload(report), report.passed


async def production_candidate_payload_for_cli(
    settings: PlatformSettings,
    *,
    command: str,
    profile: str,
    as_of: datetime,
    backup_manifest: Path | None,
    component: str,
    check_broker: bool,
    signal_sources: tuple[str, ...],
    primary_signal_name: str,
    primary_signal_type: str,
    campaign_max_age_days: int | None,
    clean_live_days: int,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None,
    broker_smoke: BrokerSmokeObservation | None,
) -> tuple[dict[str, Any], bool]:
    from quant_platform.services.governance_service.llm_live_startup import (
        write_llm_live_startup_assertion,
    )
    from quant_platform.services.governance_service.production_candidate import (
        production_candidate_payload,
    )

    report = await production_candidate_report_for_cli(
        settings,
        profile=profile,
        as_of=as_of,
        backup_manifest=backup_manifest,
        component=component,
        check_broker=check_broker,
        signal_sources=signal_sources,
        primary_signal_name=primary_signal_name,
        primary_signal_type=primary_signal_type,
        campaign_max_age_days=campaign_max_age_days,
        clean_live_days=clean_live_days,
        instrument_contracts=instrument_contracts,
        soak_report=soak_report,
        current_broker_smoke=broker_smoke,
    )
    payload = production_candidate_payload(report)
    if (
        command == "assert"
        and report.passed
        and report.profile.value in {"live", "llm_live_rehearsal"}
        and settings.llm.live_mode_enabled
    ):
        assertion_path = write_llm_live_startup_assertion(
            settings,
            candidate_payload=payload,
            as_of=as_of,
        )
        payload["llm_live_startup_assertion"] = str(assertion_path)
    return payload, report.passed


async def production_candidate_diagnostics_for_cli(
    settings: PlatformSettings,
    *,
    profile: str,
    as_of: datetime,
    backup_manifest: Path | None,
    component: str,
    check_broker: bool,
    signal_sources: tuple[str, ...],
    primary_signal_name: str,
    primary_signal_type: str,
    campaign_max_age_days: int | None,
    clean_live_days: int,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None,
    broker_smoke: BrokerSmokeObservation | None,
) -> tuple[str, bool]:
    from quant_platform.services.governance_service.production_candidate.diagnostics import (
        render_production_candidate_diagnostics,
    )

    report = await production_candidate_report_for_cli(
        settings,
        profile=profile,
        as_of=as_of,
        backup_manifest=backup_manifest,
        component=component,
        check_broker=check_broker,
        signal_sources=signal_sources,
        primary_signal_name=primary_signal_name,
        primary_signal_type=primary_signal_type,
        campaign_max_age_days=campaign_max_age_days,
        clean_live_days=clean_live_days,
        instrument_contracts=instrument_contracts,
        soak_report=soak_report,
        current_broker_smoke=broker_smoke,
    )
    return render_production_candidate_diagnostics(report), report.passed


async def production_candidate_report_for_cli(
    settings: PlatformSettings,
    *,
    profile: str,
    as_of: datetime,
    backup_manifest: Path | None,
    component: str,
    check_broker: bool,
    signal_sources: tuple[str, ...],
    primary_signal_name: str,
    primary_signal_type: str,
    campaign_max_age_days: int | None,
    clean_live_days: int,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    soak_report: Path | None,
    current_broker_smoke: BrokerSmokeObservation | None,
) -> ProductionCandidateReport:
    from quant_platform.core.domain.production import ProductionProfile
    from quant_platform.services.governance_service.production_candidate import (
        build_production_candidate_report,
    )

    await verify_postgres_schema(settings)
    evidence_repo = build_performance_repository(settings.storage.postgres_dsn)
    dataset_catalog = (
        build_dataset_catalog(settings.storage.postgres_dsn)
        if settings.storage.postgres_dsn and settings.v2.require_dataset_quorum
        else None
    )
    return await build_production_candidate_report(
        settings,
        profile=ProductionProfile(profile),
        as_of=as_of,
        instrument_contracts=instrument_contracts,
        soak_report=soak_report,
        backup_manifest=backup_manifest,
        component=component,
        broker_checked=check_broker,
        current_broker_smoke=current_broker_smoke,
        signal_sources=list(signal_sources) if signal_sources else None,
        primary_signal_name=primary_signal_name or None,
        primary_signal_type=primary_signal_type or None,
        campaign_max_age_days=campaign_max_age_days,
        clean_live_days=clean_live_days,
        evidence_repository=evidence_repo,
        dataset_catalog=dataset_catalog,
        packaged_migration_head=alembic_packaged_head(),
    )


__all__ = [
    "production_candidate_diagnostics_for_cli",
    "production_candidate_payload_for_cli",
    "production_candidate_report_for_cli",
    "readiness_payload_for_cli",
]
