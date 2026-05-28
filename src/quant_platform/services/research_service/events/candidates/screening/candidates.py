"""Deterministic SEC-event reaction candidate definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import finite_feature
from quant_platform.services.research_service.events.candidates.screening.types import (
    EventCandidateSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EVENT_REACTION_SEED_CANDIDATES: tuple[EventCandidateSpec, ...] = (
    EventCandidateSpec(
        name="abnormal_volume_3d_decay",
        formula=lambda row: (
            finite_feature(row, "vol_compression")
            * abs(finite_feature(row, "short_term_reversal_5d"))
        ),
        expression="vol_compression * abs(short_term_reversal_5d)",
        thesis="Large relative volume around SEC events should proxy attention pressure.",
    ),
    EventCandidateSpec(
        name="event_gap_reversal_1d_decay",
        formula=lambda row: -finite_feature(row, "short_term_reversal_5d"),
        expression="-short_term_reversal_5d",
        thesis="Sharp one-week event reactions may partially mean-revert after disclosure.",
    ),
    EventCandidateSpec(
        name="post_event_drift_confirmation_3d_decay",
        formula=lambda row: (
            finite_feature(row, "momentum_1m") * finite_feature(row, "trend_quality_63d")
        ),
        expression="momentum_1m * trend_quality_63d",
        thesis="Short-term event drift should be more reliable when confirmed by trend quality.",
    ),
    EventCandidateSpec(
        name="filing_cadence_surprise_decay",
        formula=lambda row: (
            finite_feature(row, "distance_to_52w_high") * finite_feature(row, "vol_compression")
        ),
        expression="distance_to_52w_high * vol_compression",
        thesis="Filing attention near price extremes may carry different forward drift.",
    ),
    EventCandidateSpec(
        name="event_attention_shock_30d_decay",
        formula=lambda row: (
            finite_feature(row, "momentum_3m") * finite_feature(row, "vol_compression")
        ),
        expression="momentum_3m * vol_compression",
        thesis="Event attention should matter most when medium-term momentum is crowded.",
    ),
)

EVENT_REACTION_V2_CANDIDATES: tuple[EventCandidateSpec, ...] = (
    EventCandidateSpec(
        name="event_reaction_v2_sec_density_price_reversal_21d",
        formula=lambda row: _event_density(row) * -finite_feature(row, "short_term_reversal_5d"),
        expression="sec_event_density_21d * -short_term_reversal_5d",
        thesis=(
            "Dense recent SEC-event cadence with a sharp price reaction can identify "
            "short-horizon overreaction without changing candidate gates."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_attention_gap_reversal_21d",
        formula=lambda row: (
            _event_density(row)
            * finite_feature(row, "vol_compression")
            * -finite_feature(row, "short_term_reversal_5d")
        ),
        expression="sec_event_density_21d * vol_compression * -short_term_reversal_5d",
        thesis=(
            "SEC-event attention under compressed volatility should make same-week "
            "price gaps more auditable as reversal candidates."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_post_event_drift_quality_21d",
        formula=lambda row: (
            _event_density(row)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "trend_quality_63d")
        ),
        expression="sec_event_density_21d * momentum_1m * trend_quality_63d",
        thesis=(
            "Post-event drift should be more stable when recent SEC-event density is "
            "confirmed by short-term momentum and trend quality."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_extreme_attention_reversal_21d",
        formula=lambda row: (
            _event_density(row)
            * finite_feature(row, "distance_to_52w_high")
            * -finite_feature(row, "short_term_reversal_5d")
        ),
        expression="sec_event_density_21d * distance_to_52w_high * -short_term_reversal_5d",
        thesis=(
            "Price reactions near 52-week extremes after SEC events can overextend "
            "and partially mean-revert."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_crowded_medium_momentum_decay_21d",
        formula=lambda row: (
            _event_density(row)
            * finite_feature(row, "momentum_3m")
            * finite_feature(row, "vol_compression")
        ),
        expression="sec_event_density_21d * momentum_3m * vol_compression",
        thesis=(
            "Repeated SEC attention in crowded medium-term momentum names can carry "
            "a measurable 21-day reaction profile."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_1_4_momo1_extreme_21d",
        formula=lambda row: (
            _event_count_band(row, 1, 4)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "distance_to_52w_high")
        ),
        expression="sec_event_count_band_1_4d * momentum_1m * distance_to_52w_high",
        thesis=(
            "SEC event reactions one to four days old should be most informative when "
            "short-term momentum is near a 52-week extreme."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_2_5_momo1_extreme_21d",
        formula=lambda row: (
            _event_count_band(row, 2, 5)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "distance_to_52w_high")
        ),
        expression="sec_event_count_band_2_5d * momentum_1m * distance_to_52w_high",
        thesis=(
            "A two-to-five-day SEC-event cadence can confirm that near-term price "
            "pressure around extremes is not only same-day filing noise."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_3_5_momo1_medium_momo_21d",
        formula=lambda row: (
            _event_count_band(row, 3, 5)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "momentum_12m_1m")
        ),
        expression="sec_event_count_band_3_5d * momentum_1m * momentum_12m_1m",
        thesis=(
            "SEC-event pressure three to five days old should be more durable when "
            "recent momentum agrees with the slower momentum context."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_3_6_momo1_reversal_21d",
        formula=lambda row: (
            -_event_count_band(row, 3, 6)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "short_term_reversal_5d")
        ),
        expression="-sec_event_count_band_3_6d * momentum_1m * short_term_reversal_5d",
        thesis=(
            "SEC-event reactions three to six days old can mean-revert when one-month "
            "momentum and the latest weekly move are crowded in the same direction."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_7_9_momo1_momo3_21d",
        formula=lambda row: (
            _event_count_band(row, 7, 9)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "momentum_3m")
        ),
        expression="sec_event_count_band_7_9d * momentum_1m * momentum_3m",
        thesis=(
            "A one-to-two-week-old SEC-event cadence may preserve drift when one-month "
            "and three-month price context agree."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_7_9_momo1_trend_21d",
        formula=lambda row: (
            _event_count_band(row, 7, 9)
            * finite_feature(row, "momentum_1m")
            * finite_feature(row, "trend_quality_63d")
        ),
        expression="sec_event_count_band_7_9d * momentum_1m * trend_quality_63d",
        thesis=(
            "Older SEC-event reactions should be more reliable when short-term "
            "momentum is supported by trend quality."
        ),
    ),
    EventCandidateSpec(
        name="event_reaction_v2_sec_count_7_9_momo3_trend_21d",
        formula=lambda row: (
            _event_count_band(row, 7, 9)
            * finite_feature(row, "momentum_3m")
            * finite_feature(row, "trend_quality_63d")
        ),
        expression="sec_event_count_band_7_9d * momentum_3m * trend_quality_63d",
        thesis=(
            "One-to-two-week SEC-event density should be most persistent when "
            "medium-term momentum is aligned with trend quality."
        ),
    ),
)


def event_candidates_for_set(candidate_set: str) -> tuple[EventCandidateSpec, ...]:
    normalized = candidate_set.strip().lower()
    if normalized == "seed":
        return EVENT_REACTION_SEED_CANDIDATES
    if normalized in {"event-reaction-v2", "event_reaction_v2", "v2"}:
        return EVENT_REACTION_V2_CANDIDATES
    raise ValueError(f"unknown event candidate set: {candidate_set}")


def _event_density(row: Mapping[str, float]) -> float:
    density = finite_feature(row, "sec_event_density_21d")
    if density > 0.0:
        return density
    count = finite_feature(row, "sec_event_count_21d")
    if count > 0.0:
        return min(count, 3.0) / 3.0
    return 0.0


def _event_count_band(row: Mapping[str, float], start_days: int, end_days: int) -> float:
    if end_days <= start_days:
        return 0.0
    end_count = finite_feature(row, f"sec_event_count_le_{end_days}d_scaled")
    start_count = finite_feature(row, f"sec_event_count_le_{start_days}d_scaled")
    return max(0.0, end_count - start_count)
