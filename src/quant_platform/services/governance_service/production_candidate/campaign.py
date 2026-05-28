"""Research-campaign evidence checks for production-candidate gates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.application.research.evidence import (
    latest_campaign_manifest_evidence,
    validate_backtest_evidence_manifest,
)
from quant_platform.core.domain.production import PreflightCheck

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings


def alembic_packaged_head() -> str:
    """Compatibility injection hook for tests; bootstrap supplies the packaged head."""
    return ""


def _migration_head_check(
    settings: PlatformSettings,
    *,
    packaged_head: str | None = None,
) -> PreflightCheck:
    head = packaged_head if packaged_head is not None else alembic_packaged_head()
    has_dsn = bool(settings.storage.postgres_dsn.strip())
    return PreflightCheck(
        name="alembic_packaged_head_resolved",
        passed=bool(head),
        detail=f"packaged head={head} dsn_configured={has_dsn}",
        severity="warning",
    )


def _campaign_manifest_check(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    max_age_days: int | None,
) -> tuple[Path | None, Mapping[str, object] | None, PreflightCheck]:
    evidence = latest_campaign_manifest_evidence(settings.storage.object_store_root)
    if not evidence.root.exists():
        return (
            None,
            None,
            PreflightCheck(
                name="research_campaign_manifest_present",
                passed=False,
                detail=f"no campaign manifests under {evidence.root}",
                severity="error",
            ),
        )
    if evidence.payload is None:
        return (
            None,
            None,
            PreflightCheck(
                name="research_campaign_manifest_present",
                passed=False,
                detail=f"no campaign manifests under {evidence.root}",
                severity="error",
            ),
        )
    payload = evidence.payload
    path = evidence.path

    stale_days = (
        max_age_days
        if max_age_days is not None
        else max(1, settings.production.data_health_stale_after_days)
    )
    created_at_raw = str(payload.get("created_at", ""))
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except ValueError:
        return (
            path,
            payload,
            PreflightCheck(
                name="research_campaign_manifest_present",
                passed=False,
                detail=f"invalid created_at: {created_at_raw!r}",
                severity="error",
            ),
        )
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    fresh = created_at >= as_of - timedelta(days=stale_days)
    return (
        path,
        payload,
        PreflightCheck(
            name="research_campaign_manifest_present",
            passed=fresh,
            detail=(
                f"latest campaign run_id={payload.get('run_id', 'unknown')} "
                f"created_at={created_at.isoformat()} stale_after_days={stale_days}"
            ),
            severity="error",
        ),
    )


def _campaign_passed_check(payload: Mapping[str, object]) -> PreflightCheck:
    passed = bool(payload.get("passed"))
    eligibility = payload.get("eligibility")
    failed_checks: list[str] = []
    if isinstance(eligibility, Mapping):
        for item in eligibility.get("checks", []) or ():
            if isinstance(item, Mapping) and not item.get("passed", True):
                failed_checks.append(str(item.get("name", "unknown")))
    detail = (
        "campaign passed=True; checks_failed=[]"
        if passed
        else f"campaign passed=False; checks_failed={failed_checks or 'unknown'}"
    )
    return PreflightCheck(
        name="research_campaign_eligibility_passed",
        passed=passed,
        detail=detail,
        severity="error",
    )


def _campaign_weights_cap_check(
    payload: Mapping[str, object],
    settings: PlatformSettings,
) -> PreflightCheck:
    weights_raw = payload.get("paper_source_weights", {})
    weights: dict[str, float] = {}
    if isinstance(weights_raw, Mapping):
        for key, value in weights_raw.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    non_classical = sum(
        weight for source, weight in weights.items() if source not in {"classical", "primary"}
    )
    cap = float(settings.alpha.paper_max_non_classical_weight)
    passed = non_classical <= cap + 1e-9
    return PreflightCheck(
        name="research_campaign_paper_weights_within_cap",
        passed=passed,
        detail=(f"non_classical={non_classical:.4f} cap={cap:.4f} weights={weights}"),
        severity="error",
    )


def _backtest_evidence_manifest_check(
    payload: Mapping[str, object],
    settings: PlatformSettings,
) -> PreflightCheck:
    """Require fail-closed intraday backtest evidence when strict mode is enabled."""
    if not settings.backtest.require_intraday_evidence:
        return PreflightCheck(
            name="intraday_backtest_evidence_manifest_passed",
            passed=True,
            detail="disabled by QP__BACKTEST__REQUIRE_INTRADAY_EVIDENCE=false",
            severity="warning",
        )
    artifacts = payload.get("artifacts")
    evidence_path: object | None = None
    if isinstance(artifacts, Mapping):
        evidence_path = artifacts.get("backtest_evidence") or artifacts.get("evidence_manifest")
    if not evidence_path:
        return PreflightCheck(
            name="intraday_backtest_evidence_manifest_passed",
            passed=False,
            detail="campaign manifest does not reference backtest evidence",
            severity="error",
        )
    try:
        validate_backtest_evidence_manifest(str(evidence_path))
    except Exception as exc:
        return PreflightCheck(
            name="intraday_backtest_evidence_manifest_passed",
            passed=False,
            detail=str(exc),
            severity="error",
        )
    return PreflightCheck(
        name="intraday_backtest_evidence_manifest_passed",
        passed=True,
        detail=str(evidence_path),
        severity="error",
    )
