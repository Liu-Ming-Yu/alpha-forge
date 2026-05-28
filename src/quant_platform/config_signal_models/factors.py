"""Classical factor and volatility sizing settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FactorSettings(BaseModel):
    """Signal factor weights for ``LinearWeightSignalModel``.

    The ``*_weight`` fields below are hand-picked priors used only when no
    fitted manifest is configured.  In production, set
    ``fitted_weights_manifest`` to a research-campaign manifest so the live
    model uses walk-forward IC-fitted weights instead.
    """

    fitted_weights_manifest: str = Field(
        default="",
        description=(
            "Path to a research-campaign manifest (campaign_manifest.json) "
            "whose 'selected_weights' are walk-forward IC-fitted factor "
            "weights. When set, supersedes the hand-picked *_weight fields "
            "below and pins the model's expected feature_set_version. "
            "Configured via QP__FACTORS__FITTED_WEIGHTS_MANIFEST."
        ),
    )

    momentum_1m_weight: float = 0.20
    momentum_3m_weight: float = 0.25
    momentum_12m_1m_weight: float = 0.35
    vol_compression_weight: float = 0.10
    short_term_reversal_5d_weight: float = 0.03
    trend_quality_63d_weight: float = 0.04
    distance_to_52w_high_weight: float = 0.03


class VolSizingSettings(BaseModel):
    """Volatility-targeted position sizing parameters."""

    enabled: bool = False
    vol_target_annualized: float = 0.15
    min_vol_floor: float = 0.05
