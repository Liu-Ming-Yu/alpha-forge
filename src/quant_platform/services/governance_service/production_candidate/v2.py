"""V2 live-path and dataset-quorum checks for production candidates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import PreflightCheck, ProductionProfile

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import DatasetCatalog


def _v2_live_path_checks(
    settings: PlatformSettings,
    profile: ProductionProfile,
) -> list[PreflightCheck]:
    """Live promotion requires V2 account orchestration to be wired."""
    if profile != ProductionProfile.LIVE:
        return []
    return [
        PreflightCheck(
            name="v2_account_orchestrator_is_live_submitter",
            passed=settings.v2.enabled and settings.v2.account_orchestrator_enabled,
            detail=(
                f"v2.enabled={settings.v2.enabled} "
                f"v2.account_orchestrator_enabled={settings.v2.account_orchestrator_enabled}"
            ),
            severity="error",
        ),
        PreflightCheck(
            name="v2_dataset_quorum_for_live",
            passed=(
                settings.v2.require_dataset_quorum and bool(settings.v2.third_eod_vendor.strip())
            ),
            detail=(
                f"require_dataset_quorum={settings.v2.require_dataset_quorum} "
                f"third_eod_vendor={settings.v2.third_eod_vendor!r}"
            ),
            severity="error",
        ),
    ]


async def _v2_dataset_quorum_evidence_check(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    profile: ProductionProfile,
    dataset_kind: str = "bars_eod",
    dataset_catalog: DatasetCatalog | None = None,
) -> PreflightCheck:
    """Require fresh persisted vendor-quorum evidence in addition to flags.

    Configuration alone (``QP__V2__REQUIRE_DATASET_QUORUM`` and
    ``QP__V2__THIRD_EOD_VENDOR``) is not sufficient: live mode must see
    a recent :class:`DatasetQuorumEvidence` row in the durable catalog
    whose ``passed`` flag is true and whose ``as_of`` timestamp falls
    inside ``settings.production.data_health_stale_after_days``.
    """
    severity = "error" if profile == ProductionProfile.LIVE else "warning"
    if profile != ProductionProfile.LIVE and not settings.backtest.require_intraday_evidence:
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=True,
            detail=(
                "paper profile skips durable quorum query unless "
                "QP__BACKTEST__REQUIRE_INTRADAY_EVIDENCE=true"
            ),
            severity=severity,
        )
    if profile == ProductionProfile.LIVE and not (
        settings.v2.enabled and settings.v2.account_orchestrator_enabled
    ):
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail="skipped durable quorum query because V2 account orchestration is disabled",
            severity=severity,
        )
    if not settings.storage.postgres_dsn.strip():
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail="postgres not configured; dataset quorum evidence cannot be persisted",
            severity=severity,
        )
    try:
        import asyncio

        if dataset_catalog is None:
            return PreflightCheck(
                name="v2_dataset_quorum_evidence_fresh",
                passed=False,
                detail="dataset catalog is not configured",
                severity=severity,
            )
        evidence = await asyncio.wait_for(
            dataset_catalog.latest_quorum_evidence(dataset_kind, as_of=as_of),
            timeout=5.0,
        )
    except Exception as exc:
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail=f"dataset quorum query failed: {exc}",
            severity=severity,
        )
    if evidence is None:
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail=f"no persisted quorum evidence for dataset_kind={dataset_kind!r}",
            severity=severity,
        )
    stale_after = timedelta(days=max(1, settings.production.data_health_stale_after_days))
    if evidence.as_of < as_of - stale_after:
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail=(
                f"latest quorum evidence at {evidence.as_of.isoformat()} is older than "
                f"{stale_after.days} days"
            ),
            severity=severity,
        )
    if not evidence.passed:
        return PreflightCheck(
            name="v2_dataset_quorum_evidence_fresh",
            passed=False,
            detail=(
                f"latest quorum evidence at {evidence.as_of.isoformat()} did not pass "
                f"(vendors={evidence.vendors!r})"
            ),
            severity=severity,
        )
    return PreflightCheck(
        name="v2_dataset_quorum_evidence_fresh",
        passed=True,
        detail=(
            f"vendors={evidence.vendors!r} as_of={evidence.as_of.isoformat()} "
            f"max_bps={evidence.details.get('max_disagreement_bps')}"
        ),
        severity=severity,
    )
