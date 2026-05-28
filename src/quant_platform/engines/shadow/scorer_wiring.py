"""Wire optional shadow scoring components for engine sessions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import structlog

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        FeatureRepository,
        PredictionEvidenceRepository,
        SignalContributionRepository,
    )
    from quant_platform.engines.framework.types import RunMode
    from quant_platform.engines.shadow.boosting_cycle import ShadowBoostingScorer
    from quant_platform.engines.shadow.text_cycle import ShadowTextCycleScorer


class ShadowScorerSession(Protocol):
    feature_repo: FeatureRepository
    signal_contribution_repo: SignalContributionRepository | None


def build_shadow_text_scorer(
    *,
    settings: PlatformSettings,
    session: ShadowScorerSession,
) -> ShadowTextCycleScorer | None:
    """Build the LLM shadow text scorer when enabled."""
    if not settings.llm.shadow_mode_enabled:
        return None

    from quant_platform.services.governance_service.llm_live_startup import (
        llm_extraction_artifact_root,
    )
    from quant_platform.services.research_service.text.features import (
        LLMTextFeatureExtractor,
    )
    from quant_platform.services.research_service.text.shadow.scorer import ShadowTextScorer

    extractor = LLMTextFeatureExtractor(
        provider=settings.llm.provider,
        model=settings.llm.model,
        prompt_version=settings.llm.text_prompt_version,
        max_tokens=settings.llm.max_tokens,
        timeout_seconds=settings.llm.timeout_seconds,
        deepseek_base_url=settings.llm.deepseek_base_url,
        artifact_root=llm_extraction_artifact_root(settings),
        replay_only=settings.llm.live_mode_enabled and settings.llm.replay_only_live,
        max_request_latency_seconds=settings.llm.max_request_latency_seconds,
        max_daily_calls=settings.llm.max_daily_calls,
        max_daily_estimated_cost_usd=settings.llm.max_daily_estimated_cost_usd,
        estimated_cost_per_call_usd=settings.llm.estimated_cost_per_call_usd,
    )
    scorer = ShadowTextScorer(
        extractor=extractor,
        feature_repo=session.feature_repo,
        contribution_repo=session.signal_contribution_repo,
        prediction_evidence_repo=cast(
            "PredictionEvidenceRepository | None",
            getattr(session, "performance_repo", None),
        ),
    )
    log.info(
        "engine_runner.shadow_text_scorer.enabled",
        model=settings.llm.model,
        prompt_version=settings.llm.text_prompt_version,
    )
    return scorer


def build_shadow_boosting_scorer(
    *,
    settings: PlatformSettings,
    run_mode: RunMode,
    session: ShadowScorerSession,
) -> ShadowBoostingScorer | None:
    """Build the boosted-tree shadow scorer when enabled for shadow mode."""
    if not settings.boosting.enabled:
        return None

    if getattr(run_mode, "value", run_mode) != "shadow":
        log.warning(
            "engine_runner.shadow_boosting.disabled_non_shadow_mode",
            mode=getattr(run_mode, "value", run_mode),
        )
        return None

    if not settings.boosting.artifact_manifest:
        raise ValueError(
            "QP__BOOSTING__ARTIFACT_MANIFEST is required when QP__BOOSTING__ENABLED=true"
        )

    from quant_platform.services.research_service.boosting import (
        ShadowBoostingScorer,
        XGBoostRankSignalModel,
    )

    boosting_model = XGBoostRankSignalModel(
        Path(settings.boosting.artifact_manifest),
        device=settings.boosting.device,
        require_gpu=settings.boosting.require_gpu,
    )
    scorer = ShadowBoostingScorer(
        model=boosting_model,
        artifact_root=Path(settings.boosting.shadow_artifact_root),
        contribution_repo=session.signal_contribution_repo,
        prediction_evidence_repo=cast(
            "PredictionEvidenceRepository | None",
            getattr(session, "performance_repo", None),
        ),
    )
    log.info(
        "engine_runner.shadow_boosting.enabled",
        model_version=boosting_model.model_version,
        feature_schema_hash=boosting_model.feature_schema_hash,
        device=boosting_model.device,
    )
    return scorer
