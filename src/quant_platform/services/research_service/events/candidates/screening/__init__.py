"""Read-only screening for SEC-event/price-reaction paper alpha candidates."""

from __future__ import annotations

from quant_platform.services.research_service.events.candidates.screening.artifacts import (
    render_event_candidate_screen_report,
    write_event_candidate_family_artifacts,
)
from quant_platform.services.research_service.events.candidates.screening.candidates import (
    EVENT_REACTION_SEED_CANDIDATES,
    EVENT_REACTION_V2_CANDIDATES,
    event_candidates_for_set,
)
from quant_platform.services.research_service.events.candidates.screening.context import (
    event_context_features,
    events_by_instrument_from_manifest,
)
from quant_platform.services.research_service.events.candidates.screening.screen import (
    build_event_candidate_screen,
)
from quant_platform.services.research_service.events.candidates.screening.types import (
    EVENT_REACTION_FEATURE_SET_VERSION,
    EVENT_REACTION_V2_FEATURE_SET_VERSION,
    EventCandidateScreenThresholds,
    EventCandidateSpec,
)

__all__ = [
    "EVENT_REACTION_FEATURE_SET_VERSION",
    "EVENT_REACTION_V2_CANDIDATES",
    "EVENT_REACTION_V2_FEATURE_SET_VERSION",
    "EVENT_REACTION_SEED_CANDIDATES",
    "EventCandidateScreenThresholds",
    "EventCandidateSpec",
    "build_event_candidate_screen",
    "event_context_features",
    "event_candidates_for_set",
    "events_by_instrument_from_manifest",
    "render_event_candidate_screen_report",
    "write_event_candidate_family_artifacts",
]
