"""Shared event-candidate types and constants."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from quant_platform.services.research_service.campaigns.screening.common import (
    CandidateScreenThresholds,
)

EventFormula = Callable[[Mapping[str, float]], float]
EventCandidateScreenThresholds = CandidateScreenThresholds
EVENT_REACTION_FEATURE_SET_VERSION = "paper-alpha-event-reaction-v1"
EVENT_REACTION_V2_FEATURE_SET_VERSION = "paper-alpha-event-reaction-v2"


@dataclass(frozen=True)
class EventCandidateSpec:
    """One prospective event/price-reaction formula for diagnostic screening."""

    name: str
    formula: EventFormula
    expression: str
    thesis: str
