"""Text candidate screening package."""

from __future__ import annotations

from quant_platform.services.research_service.text.candidates.catalog import (
    TEXT_CATALYST_V10_PROMOTED_CANDIDATES,
    V10_ALPHA_QUALITY_TEXT_CANDIDATES,
    TextCandidateSpec,
    build_text_aggregate_features,
    text_candidates_for_set,
)
from quant_platform.services.research_service.text.candidates.screening.artifacts import (
    write_text_candidate_family_artifacts,
)
from quant_platform.services.research_service.text.candidates.screening.screen import (
    TextCandidateScreenThresholds,
    build_text_candidate_screen,
    render_text_candidate_screen_report,
)

__all__ = [
    "TEXT_CATALYST_V10_PROMOTED_CANDIDATES",
    "TextCandidateScreenThresholds",
    "TextCandidateSpec",
    "V10_ALPHA_QUALITY_TEXT_CANDIDATES",
    "build_text_aggregate_features",
    "build_text_candidate_screen",
    "render_text_candidate_screen_report",
    "text_candidates_for_set",
    "write_text_candidate_family_artifacts",
]
