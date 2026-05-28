"""Campaign-level feature audit orchestration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.features.governance_payloads import (
    feature_audit_result_payload,
    manifest_path,
)
from quant_platform.services.research_service.feature_quality.audit.feature_audit import (
    FeatureAuditRunner,
    FeatureAuditThresholds,
    generated_shadow_feature_definition,
    load_feature_definition,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.features.governance_requests import (
        CampaignFeatureAuditRequest,
    )
    from quant_platform.core.contracts import ArtifactStore, FeatureAuditRepository
    from quant_platform.core.domain.research import FeatureDefinition


async def audit_campaign_features(
    *,
    request: CampaignFeatureAuditRequest,
    object_store_root: Path,
    repository: FeatureAuditRepository | None,
    artifact_store: ArtifactStore | None,
) -> list[dict[str, object]]:
    if request.mode == "off":
        return []

    feature_names = sorted(
        {
            name
            for sample in request.samples
            for name in getattr(sample, "features", {})
            if not str(name).startswith("_")
        }
    )
    candidate_names = (
        sorted({str(name) for name in request.candidate_feature_names})
        if request.candidate_feature_names is not None
        else feature_names
    )
    candidate_name_set = (
        set(candidate_names) if request.candidate_feature_names is not None else set()
    )
    rows: list[dict[str, object]] = []
    for feature_name in candidate_names:
        definition = campaign_feature_definition(
            feature_name=feature_name,
            request=request,
            rows=rows,
        )
        if definition is None:
            continue
        baseline = [
            name
            for name in feature_names
            if name != feature_name and name not in candidate_name_set
        ]
        manifest = FeatureAuditRunner(
            thresholds=FeatureAuditThresholds(min_daily_groups=252),
            slippage_bps_per_turnover=request.slippage_bps_per_turnover,
            baseline_features=baseline,
            artifact_store=artifact_store,
        ).run(
            feature=definition,
            samples=request.samples,
            feature_set_version=request.feature_set_version,
            output_root=object_store_root,
        )
        path = manifest_path(object_store_root, definition, manifest.audit_id)
        result = manifest.to_result(str(path))
        if repository is not None:
            await repository.save_feature_audit(result)
        payload = feature_audit_result_payload(result)
        payload["manifest"] = str(path)
        rows.append(payload)
    return rows


def campaign_feature_definition(
    *,
    feature_name: str,
    request: CampaignFeatureAuditRequest,
    rows: list[dict[str, object]],
) -> FeatureDefinition | None:
    card_path = (
        request.feature_card_dir / f"{feature_name}.json" if request.feature_card_dir else None
    )
    if card_path is not None and card_path.is_file():
        return load_feature_definition(card_path)
    if request.mode == "paper":
        rows.append(
            {
                "feature_name": feature_name,
                "passed": False,
                "blockers": [f"missing feature card: {card_path}"],
            }
        )
        return None
    return generated_shadow_feature_definition(
        feature_name=feature_name,
        feature_set_version=request.feature_set_version,
        horizon_days=request.horizon_days,
    )
