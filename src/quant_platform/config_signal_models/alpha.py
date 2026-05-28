"""Governed alpha-promotion and ensemble settings."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AlphaSettings(BaseModel):
    """Governed alpha-promotion and ensemble configuration."""

    ensemble_mode: Literal["shadow", "paper", "live"] = "shadow"
    source_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "classical": 0.70,
            "xgboost": 0.15,
            "text": 0.05,
            "event": 0.05,
            "intraday": 0.05,
        },
        description=(
            "Ensemble blend weights for classical, xgboost, text, event, and intraday sources."
        ),
    )
    promoted_feature_set_version: str = Field(
        default="paper-alpha-composite-v1",
        description=(
            "Governed paper ensemble feature_set_version used to validate composite "
            "text/event/intraday feature audits before promoted paper scoring."
        ),
    )
    event_feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "event_reaction_v2_sec_count_3_5_momo1_medium_momo_21d": 0.40,
            "event_reaction_v2_sec_count_7_9_momo1_momo3_21d": 0.35,
            "event_reaction_v2_sec_count_7_9_momo3_trend_21d": 0.25,
        },
        description="Weights for promoted paper event-reaction features.",
    )
    event_feature_set_version: str = Field(
        default="paper-alpha-event-reaction-v2",
        description="Audited feature_set_version for promoted paper event features.",
    )
    event_feature_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional feature-version map for promoted event feature admission. When empty, "
            "each configured event feature weight uses event_feature_set_version."
        ),
    )
    intraday_feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "intraday_v2_range_vwap_band_composite_21d": 0.35,
            "intraday_v2_short_range_vwap_opening_composite_21d": 0.35,
            "intraday_v2_signed_range_expansion_band_2_3_close_pressure_21d": 0.30,
        },
        description="Weights for promoted paper intraday microstructure features.",
    )
    intraday_feature_set_version: str = Field(
        default="paper-alpha-intraday-microstructure-v2",
        description="Audited feature_set_version for promoted paper intraday features.",
    )
    intraday_feature_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional feature-version map for promoted intraday feature admission. When empty, "
            "each configured intraday feature weight uses intraday_feature_set_version."
        ),
    )
    max_non_classical_weight: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Initial live cap for combined non-classical alpha influence.",
    )
    paper_max_non_classical_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Aggressive paper-only cap for combined XGBoost/text influence.",
    )
    fail_closed_on_promoted_source_error: bool = True
    require_promotion_gate: bool = True
    live_ramp_initial: Decimal = Decimal("0.01")
    live_ramp_after_20d: Decimal = Decimal("0.10")
    live_ramp_after_60d: Decimal = Decimal("0.20")

    @field_validator("source_weights")
    @classmethod
    def validate_source_weights(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = {"classical", "xgboost", "text", "event", "intraday"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown alpha source weights: {', '.join(unknown)}")
        if "classical" not in value:
            raise ValueError("alpha.source_weights must include classical")
        total = sum(float(v) for v in value.values())
        if total <= 0:
            raise ValueError("alpha.source_weights must sum to a positive value")
        for name, weight in value.items():
            if weight < 0:
                raise ValueError(f"alpha source weight for {name} must be non-negative")
        return {str(k): float(v) for k, v in value.items()}
