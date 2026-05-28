"""Live-LLM evidence check orchestration for production-candidate gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    ForecastEvidence,
    PreflightCheck,
    ProductionProfile,
)
from quant_platform.services.governance_service.llm_live_startup.feature_card_checks import (
    text_feature_admission_check,
    text_feature_card_checks,
    text_feature_card_dir_deployment_checks,
)
from quant_platform.services.governance_service.llm_live_startup.manifest_checks import (
    load_text_model_manifest,
    text_manifest_policy_checks,
)
from quant_platform.services.governance_service.llm_live_startup.prediction_checks import (
    prediction_schema_hash_check,
    prediction_source_horizon_check,
)
from quant_platform.services.governance_service.llm_live_startup.runtime_limits import (
    build_runtime_limit_checks,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings


def build_llm_live_evidence_checks(
    settings: PlatformSettings,
    *,
    as_of: datetime,
    forecast_evidence: ForecastEvidence | None = None,
    profile: ProductionProfile = ProductionProfile.LIVE,
) -> list[PreflightCheck]:
    """Return synchronous live-LLM evidence checks for production-candidate gates."""
    if not settings.llm.live_mode_enabled:
        return []

    checks = build_runtime_limit_checks(settings)
    manifest_evidence, manifest_check = load_text_model_manifest(settings)
    checks.append(manifest_check)
    if manifest_evidence is None:
        return checks

    manifest = manifest_evidence.payload
    try:
        checks.extend(text_manifest_policy_checks(settings, manifest=manifest, as_of=as_of))
    except RuntimeError as exc:
        checks.append(
            PreflightCheck(
                name="llm_live_text_manifest_fresh",
                passed=False,
                detail=str(exc),
            )
        )
    checks.extend(text_feature_card_dir_deployment_checks(settings, profile=profile))
    checks.extend(text_feature_card_checks(settings, manifest=manifest))
    checks.append(text_feature_admission_check(settings, manifest=manifest, profile=profile))
    if forecast_evidence is not None:
        checks.append(prediction_source_horizon_check(forecast_evidence))
        checks.append(prediction_schema_hash_check(settings, forecast_evidence))
    return checks
