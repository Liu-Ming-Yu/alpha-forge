"""Current text alpha-quality formula helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import finite_feature

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.services.research_service.text.candidates.catalog.types import (
        CandidateFormula,
    )


def _aggregate_field(field: str, suffix: str = "decayed_mean") -> str:
    return f"{field}_{suffix}_21d"


def _coverage_gate(text: Mapping[str, float], threshold: float) -> float:
    coverage = finite_feature(text, "text_cross_section_coverage_21d")
    return 1.0 if coverage >= threshold else 0.0


def _v10_abs_text_tone_minus_vol_tone_formula(
    *,
    coverage_threshold: float,
    vol_multiplier: float,
) -> CandidateFormula:
    def formula(text: Mapping[str, float], sample: Mapping[str, float], _decay: float) -> float:
        tone = finite_feature(text, _aggregate_field("text_sentiment"))
        coverage = _coverage_gate(text, coverage_threshold)
        return abs(tone) * coverage - vol_multiplier * tone * finite_feature(
            sample, "vol_compression"
        )

    return formula


def _v10_abs_text_specificity_context_formula(context_field: str) -> CandidateFormula:
    def formula(
        text: Mapping[str, float],
        sample: Mapping[str, float],
        _decay: float,
        *,
        raw_context_field: str = context_field,
    ) -> float:
        tone = finite_feature(text, _aggregate_field("text_sentiment"))
        specificity = finite_feature(text, _aggregate_field("disclosure_specificity"))
        context = finite_feature(text, _aggregate_field(raw_context_field))
        vol_pos = max(finite_feature(sample, "vol_compression"), 0.0)
        coverage = _coverage_gate(text, 0.50)
        return (
            abs(tone) * coverage - 0.5 * specificity * vol_pos * coverage - 3.0 * context * vol_pos
        )

    return formula


def _v10_specificity_surprise_residual_formula() -> CandidateFormula:
    def formula(text: Mapping[str, float], _sample: Mapping[str, float], _decay: float) -> float:
        event_surprise = finite_feature(text, "event_surprise_decayed_mean_21d")
        specificity = finite_feature(text, "disclosure_specificity_decayed_mean_21d")
        forward_outlook = max(finite_feature(text, "forward_outlook_decayed_mean_21d"), 0.0)
        return -event_surprise * specificity - 0.25 * forward_outlook

    return formula


def _v10_outlook_operating_disagreement_formula() -> CandidateFormula:
    def formula(text: Mapping[str, float], _sample: Mapping[str, float], _decay: float) -> float:
        operating_quality = finite_feature(text, "operating_quality_decayed_mean_21d")
        forward_outlook = finite_feature(text, "forward_outlook_decayed_mean_21d")
        specificity = finite_feature(text, "disclosure_specificity_decayed_mean_21d")
        return abs(operating_quality - forward_outlook) * specificity * _coverage_gate(text, 0.40)

    return formula


def _v10_risk_pressure_reversal_formula() -> CandidateFormula:
    def formula(text: Mapping[str, float], _sample: Mapping[str, float], _decay: float) -> float:
        tone = finite_feature(text, "text_sentiment_decayed_mean_21d")
        risk_pressure = finite_feature(text, "risk_pressure_decayed_mean_21d")
        return -tone * max(0.0, 1.0 - risk_pressure) * _coverage_gate(text, 0.40)

    return formula


def _v10_tone_change_confirmation_formula() -> CandidateFormula:
    def formula(text: Mapping[str, float], _sample: Mapping[str, float], _decay: float) -> float:
        recent_sum = finite_feature(text, "text_sentiment_decayed_sum_clipped_7d")
        full_sum = finite_feature(text, "text_sentiment_decayed_sum_clipped_42d")
        consistency = finite_feature(text, "text_sentiment_sign_consistency_21d")
        return (recent_sum - full_sum) * consistency

    return formula


def _v10_coverage_event_density_composite_formula() -> CandidateFormula:
    def formula(text: Mapping[str, float], _sample: Mapping[str, float], _decay: float) -> float:
        tone = finite_feature(text, "text_sentiment_decayed_mean_21d")
        event_scale = min(finite_feature(text, "text_event_count_21d"), 3.0) / 3.0
        return -tone * _coverage_gate(text, 0.40) * event_scale

    return formula
