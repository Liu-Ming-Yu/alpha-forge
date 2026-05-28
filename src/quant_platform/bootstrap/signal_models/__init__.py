"""Signal-model composition for runtime sessions.

The public session facade is synchronous, so this module performs only
synchronous artifact checks. Durable repository checks belong in async
application/bootstrap paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.bootstrap.signal_models.admission import (
    assert_promoted_alpha_sources_configured,
    assert_promoted_boosting_features_admitted,
    load_boosting_manifest_policy_payload,
    promoted_weight,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import SignalModel


def build_default_signal_model(settings: PlatformSettings) -> LinearWeightSignalModel:
    """Build the classical signal model.

    When ``factors.fitted_weights_manifest`` is set, the model uses the
    walk-forward IC-fitted weights from that campaign manifest (the data-driven
    replacement for hand-picked priors).  Otherwise it falls back to the
    hand-picked ``FactorSettings`` defaults.
    """

    fs = settings.factors
    if fs.fitted_weights_manifest:
        from quant_platform.bootstrap.signal_models.classical_manifest import (
            load_classical_signal_model,
        )

        return load_classical_signal_model(Path(fs.fitted_weights_manifest))

    weights = {}
    if fs.momentum_1m_weight:
        weights["momentum_1m"] = fs.momentum_1m_weight
    if fs.momentum_3m_weight:
        weights["momentum_3m"] = fs.momentum_3m_weight
    if fs.momentum_12m_1m_weight:
        weights["momentum_12m_1m"] = fs.momentum_12m_1m_weight
    if fs.vol_compression_weight:
        weights["vol_compression"] = fs.vol_compression_weight
    if fs.short_term_reversal_5d_weight:
        weights["short_term_reversal_5d"] = fs.short_term_reversal_5d_weight
    if fs.trend_quality_63d_weight:
        weights["trend_quality_63d"] = fs.trend_quality_63d_weight
    if fs.distance_to_52w_high_weight:
        weights["distance_to_52w_high"] = fs.distance_to_52w_high_weight
    return LinearWeightSignalModel(weights)


def build_default_primary_signal_model(settings: PlatformSettings) -> SignalModel:
    """Build the runtime primary model and enforce promoted-source admission."""

    classical = build_default_signal_model(settings)
    if settings.alpha.ensemble_mode == "shadow":
        return classical
    assert_promoted_alpha_sources_configured(settings)

    from quant_platform.services.signal_service.ensemble import build_default_ensemble

    xgboost_model = None
    if settings.boosting.artifact_manifest:
        manifest_path = Path(settings.boosting.artifact_manifest)
        manifest = load_boosting_manifest_policy_payload(manifest_path)
        if promoted_weight(settings, "xgboost") > 0 and settings.alpha.require_promotion_gate:
            assert_promoted_boosting_features_admitted(settings, manifest)

        from quant_platform.services.research_service.boosting import XGBoostRankSignalModel

        xgboost_model = XGBoostRankSignalModel(
            manifest_path,
            device=settings.boosting.device,
            require_gpu=settings.boosting.require_gpu,
        )

    text_model = None
    text_feature_weights = _positive_weight_mapping(settings.llm.text_feature_weights)
    if settings.llm.live_mode_enabled or settings.alpha.source_weights.get("text", 0.0) > 0:
        text_model = LinearWeightSignalModel(
            text_feature_weights,
            model_version=f"{settings.llm.text_feature_set_version}:text",
            strict_missing=True,
            expected_feature_set_version=settings.llm.text_feature_set_version,
        )

    event_model = None
    event_feature_weights = _positive_weight_mapping(settings.alpha.event_feature_weights)
    if settings.alpha.source_weights.get("event", 0.0) > 0:
        event_model = LinearWeightSignalModel(
            event_feature_weights,
            model_version=f"event-{settings.alpha.event_feature_set_version}",
        )

    intraday_model = None
    intraday_feature_weights = _positive_weight_mapping(settings.alpha.intraday_feature_weights)
    if settings.alpha.source_weights.get("intraday", 0.0) > 0:
        intraday_model = LinearWeightSignalModel(
            intraday_feature_weights,
            model_version=f"intraday-{settings.alpha.intraday_feature_set_version}",
        )

    return build_default_ensemble(
        classical_model=classical,
        text_model=text_model,
        xgboost_model=xgboost_model,
        event_model=event_model,
        intraday_model=intraday_model,
        source_weights=settings.alpha.source_weights,
        mode=settings.alpha.ensemble_mode,
        max_non_classical_weight=alpha_non_classical_cap(settings),
        fail_closed=settings.alpha.fail_closed_on_promoted_source_error,
        text_required_features=set(text_feature_weights),
        required_features_by_source={
            "text": set(text_feature_weights),
            "event": set(event_feature_weights),
            "intraday": set(intraday_feature_weights),
        },
        model_version=f"ensemble-{settings.alpha.ensemble_mode}-v1",
    )


def alpha_non_classical_cap(settings: PlatformSettings) -> float:
    if settings.alpha.ensemble_mode == "paper":
        return settings.alpha.paper_max_non_classical_weight
    return settings.alpha.max_non_classical_weight


def _positive_weight_mapping(weights: dict[str, float]) -> dict[str, float]:
    return {str(name): float(weight) for name, weight in weights.items() if float(weight) != 0.0}


__all__ = [
    "alpha_non_classical_cap",
    "assert_promoted_alpha_sources_configured",
    "build_default_primary_signal_model",
    "build_default_signal_model",
]
