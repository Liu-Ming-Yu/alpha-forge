"""Current deterministic text candidate catalog."""

from __future__ import annotations

from quant_platform.services.research_service.text.candidates.catalog.formulas import (
    _v10_abs_text_specificity_context_formula,
    _v10_abs_text_tone_minus_vol_tone_formula,
    _v10_coverage_event_density_composite_formula,
    _v10_outlook_operating_disagreement_formula,
    _v10_risk_pressure_reversal_formula,
    _v10_specificity_surprise_residual_formula,
    _v10_tone_change_confirmation_formula,
)
from quant_platform.services.research_service.text.candidates.catalog.types import TextCandidateSpec


def v10_alpha_quality_text_candidates() -> tuple[TextCandidateSpec, ...]:
    """Deterministic v10 formulas using multi-window text quality context."""
    return (
        TextCandidateSpec(
            name="v10_stability_abs_text_specificity_event_surprise_21d",
            expression=(
                "abs(text_sentiment_decayed_mean_21d) * "
                "I(text_cross_section_coverage_21d >= 0.50) "
                "- 0.5 * disclosure_specificity_decayed_mean_21d "
                "* max(vol_compression, 0) "
                "- 3 * event_surprise_decayed_mean_21d * max(vol_compression, 0)"
            ),
            formula=_v10_abs_text_specificity_context_formula("event_surprise"),
            thesis=(
                "Text specificity and event surprise should be evaluated "
                "under current coverage gates."
            ),
        ),
        TextCandidateSpec(
            name="v10_stability_abs_text_specificity_forward_outlook_21d",
            expression=(
                "abs(text_sentiment_decayed_mean_21d) * "
                "I(text_cross_section_coverage_21d >= 0.50) "
                "- 0.5 * disclosure_specificity_decayed_mean_21d "
                "* max(vol_compression, 0) "
                "- 3 * forward_outlook_decayed_mean_21d * max(vol_compression, 0)"
            ),
            formula=_v10_abs_text_specificity_context_formula("forward_outlook"),
            thesis=(
                "Text specificity and forward outlook should remain robust "
                "under current coverage gates."
            ),
        ),
        TextCandidateSpec(
            name="v10_stability_abs_text_tone_cov40_minus_vol_tone_21d",
            expression=(
                "abs(text_sentiment_decayed_mean_21d) * "
                "I(text_cross_section_coverage_21d >= 0.40) "
                "- 2 * text_sentiment_decayed_mean_21d * vol_compression"
            ),
            formula=_v10_abs_text_tone_minus_vol_tone_formula(
                coverage_threshold=0.40,
                vol_multiplier=2.0,
            ),
            thesis=(
                "Coverage-gated absolute tone should be penalized when "
                "volatility confirms crowded tone."
            ),
        ),
        TextCandidateSpec(
            name="v10_specificity_surprise_residual_21d",
            expression=(
                "-event_surprise_decayed_mean_21d * "
                "disclosure_specificity_decayed_mean_21d "
                "- 0.25 * max(forward_outlook_decayed_mean_21d, 0)"
            ),
            formula=_v10_specificity_surprise_residual_formula(),
            thesis=(
                "Specific primary filing surprise should retain alpha after "
                "discounting optimistic outlook."
            ),
        ),
        TextCandidateSpec(
            name="v10_outlook_operating_disagreement_21d",
            expression=(
                "abs(operating_quality_decayed_mean_21d - "
                "forward_outlook_decayed_mean_21d) * "
                "disclosure_specificity_decayed_mean_21d * "
                "I(text_cross_section_coverage_21d >= 0.40)"
            ),
            formula=_v10_outlook_operating_disagreement_formula(),
            thesis=(
                "Specific gaps between operating quality and outlook can "
                "identify repricing pressure."
            ),
        ),
        TextCandidateSpec(
            name="v10_risk_pressure_reversal_21d",
            expression=(
                "-text_sentiment_decayed_mean_21d * "
                "(1 - risk_pressure_decayed_mean_21d) * "
                "I(text_cross_section_coverage_21d >= 0.40)"
            ),
            formula=_v10_risk_pressure_reversal_formula(),
            thesis=(
                "Negative filing tone is cleaner when risk pressure is not "
                "dominating the disclosure."
            ),
        ),
        TextCandidateSpec(
            name="v10_tone_change_confirmation_7_42d",
            expression=(
                "(text_sentiment_decayed_sum_clipped_7d - "
                "text_sentiment_decayed_sum_clipped_42d) * "
                "text_sentiment_sign_consistency_21d"
            ),
            formula=_v10_tone_change_confirmation_formula(),
            thesis=(
                "Recent tone is more informative when it confirms the "
                "broader primary-text direction."
            ),
        ),
        TextCandidateSpec(
            name="v10_coverage_event_density_composite_21d",
            expression=(
                "-text_sentiment_decayed_mean_21d * "
                "I(text_cross_section_coverage_21d >= 0.40) * "
                "min(text_event_count_21d, 3) / 3"
            ),
            formula=_v10_coverage_event_density_composite_formula(),
            thesis=(
                "Repeated covered filing events should be more reliable "
                "than sparse single-event tone."
            ),
        ),
    )


__all__ = ["v10_alpha_quality_text_candidates"]
