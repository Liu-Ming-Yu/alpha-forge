"""Paper-alpha SEC-event/price-reaction feature materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.events.candidates.screening import (
    EVENT_REACTION_V2_CANDIDATES,
    EVENT_REACTION_V2_FEATURE_SET_VERSION,
    EventCandidateSpec,
    event_context_features,
)
from quant_platform.services.research_service.features.cross_section.cross_section import (
    FeatureBundle,
    build_feature_bundle,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.domain.market_data import MarketBar

PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION = EVENT_REACTION_V2_FEATURE_SET_VERSION
EVENT_REACTION_V2_PROMOTED_FEATURES = (
    "event_reaction_v2_sec_count_3_5_momo1_medium_momo_21d",
    "event_reaction_v2_sec_count_7_9_momo1_momo3_21d",
    "event_reaction_v2_sec_count_7_9_momo3_trend_21d",
)
EVENT_REACTION_V2_ALPHA_FEATURES = EVENT_REACTION_V2_PROMOTED_FEATURES


def build_paper_alpha_event_reaction_v2_feature_bundle(
    bar_data: Mapping[uuid.UUID, Sequence[MarketBar]],
    *,
    as_of: datetime | None = None,
    events_by_instrument: Mapping[uuid.UUID, Sequence[datetime]] | None = None,
) -> FeatureBundle:
    """Build deterministic event-reaction v2 features from SEC event and price context."""
    close_history = {
        instrument_id: [float(bar.close) for bar in bars]
        for instrument_id, bars in bar_data.items()
    }
    classical = build_feature_bundle(close_history)
    event_context_as_of = as_of or max(
        (bar.timestamp for bars in bar_data.values() for bar in bars),
        default=None,
    )
    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for instrument_id, row in classical.alpha_features.items():
        features = dict(row)
        formula_row = dict(row)
        if event_context_as_of is not None:
            formula_row.update(
                event_context_features(
                    instrument_id=instrument_id,
                    as_of=event_context_as_of,
                    events_by_instrument=events_by_instrument or {},
                )
            )
        for candidate in _promoted_event_v2_candidates():
            features[candidate.name] = candidate.formula(formula_row)
        alpha_features[instrument_id] = features
    return FeatureBundle(
        alpha_features=alpha_features,
        vol_forecasts=dict(classical.vol_forecasts),
    )


def _promoted_event_v2_candidates() -> tuple[EventCandidateSpec, ...]:
    promoted = set(EVENT_REACTION_V2_PROMOTED_FEATURES)
    return tuple(
        candidate for candidate in EVENT_REACTION_V2_CANDIDATES if candidate.name in promoted
    )


__all__ = [
    "EVENT_REACTION_V2_ALPHA_FEATURES",
    "EVENT_REACTION_V2_PROMOTED_FEATURES",
    "PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION",
    "build_paper_alpha_event_reaction_v2_feature_bundle",
]
