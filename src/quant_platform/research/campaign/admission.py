"""Feature-audit admission orchestration for research campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.research.campaign.blocked import write_blocked_campaign_summary
from quant_platform.research.campaign.feature_audit_ops import (
    _run_campaign_feature_audits,
)
from quant_platform.services.research_service.campaigns.evaluation.feature_admission import (
    CampaignFeatureAdmission,
    FeatureAdmissionMode,
    annotate_feature_audits,
    resolve_campaign_feature_admission,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@dataclass(frozen=True)
class CampaignAdmissionResolution:
    feature_audits: list[dict[str, object]]
    admission: CampaignFeatureAdmission
    campaign_context: dict[str, object]
    blocked_payload: dict[str, object] | None = None


async def resolve_campaign_admission(
    *,
    settings: PlatformSettings,
    samples: Sequence[SupervisedAlphaSample],
    sample_build: Mapping[str, object],
    output_root: Path,
    sample_slug: str,
    feature_set_version: str,
    horizon_days: int,
    slippage_bps_per_turnover: float,
    feature_audit_mode: str,
    feature_card_dir: Path | None,
    feature_admission: FeatureAdmissionMode,
    min_admitted_features: int,
    date_policy: str,
    feature_diagnostics_path: Path | None = None,
    feature_attribution_path: Path | None = None,
) -> CampaignAdmissionResolution:
    """Run feature audits, resolve admission, and prepare blocked payloads."""
    feature_audits = await _run_campaign_feature_audits(
        settings=settings,
        samples=samples,
        feature_set_version=feature_set_version,
        horizon_days=horizon_days,
        slippage_bps_per_turnover=slippage_bps_per_turnover,
        mode=feature_audit_mode,
        feature_card_dir=feature_card_dir,
    )
    admission = resolve_campaign_feature_admission(
        samples=samples,
        feature_audits=feature_audits,
        audit_mode=feature_audit_mode,
        feature_admission=feature_admission,
        min_admitted_features=min_admitted_features,
        candidate_feature_names=_admission_candidate_feature_names(
            feature_audits,
            feature_audit_mode,
        ),
    )
    annotated_audits = annotate_feature_audits(feature_audits, admission)
    campaign_context: dict[str, object] = {
        "date_policy": date_policy,
        "sample_build": dict(sample_build),
        "feature_admission": admission.to_payload(),
    }
    if feature_diagnostics_path is not None:
        campaign_context["feature_direction_diagnostics"] = str(feature_diagnostics_path)
    if feature_attribution_path is not None:
        campaign_context["feature_failure_attribution"] = str(feature_attribution_path)
    if admission.passed:
        return CampaignAdmissionResolution(
            feature_audits=annotated_audits,
            admission=admission,
            campaign_context=campaign_context,
        )

    summary_path = write_blocked_campaign_summary(
        output_root=output_root,
        sample_slug=sample_slug,
        reason="feature admission blocked paper research campaign",
        sample_build=sample_build,
        feature_audits=annotated_audits,
        feature_admission=admission.to_payload(),
        date_policy=date_policy,
        feature_diagnostics_path=feature_diagnostics_path,
        feature_attribution_path=feature_attribution_path,
    )
    return CampaignAdmissionResolution(
        feature_audits=annotated_audits,
        admission=admission,
        campaign_context=campaign_context,
        blocked_payload={
            "passed": False,
            "reason": "feature admission blocked paper research campaign",
            "blocked_campaign_summary": str(summary_path),
            "date_policy": date_policy,
            "feature_direction_diagnostics": str(feature_diagnostics_path)
            if feature_diagnostics_path is not None
            else None,
            "feature_failure_attribution": str(feature_attribution_path)
            if feature_attribution_path is not None
            else None,
            "sample_build": dict(sample_build),
            "feature_audits": annotated_audits,
            "feature_admission": admission.to_payload(),
        },
    )


def _admission_candidate_feature_names(
    feature_audits: Sequence[Mapping[str, object]],
    feature_audit_mode: str,
) -> tuple[str, ...] | None:
    if feature_audit_mode != "paper":
        return None
    return tuple(str(row.get("feature_name")) for row in feature_audits if row.get("feature_name"))


__all__ = ["CampaignAdmissionResolution", "resolve_campaign_admission"]
