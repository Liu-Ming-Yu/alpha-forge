"""Deterministic candidate catalogs for text-derived alpha screens."""

from __future__ import annotations

from quant_platform.services.research_service.text.candidates.catalog.aggregates import (
    build_text_aggregate_features,
)
from quant_platform.services.research_service.text.candidates.catalog.fields import (
    TEXT_CATALYST_V10_PROMOTED_CANDIDATES,
)
from quant_platform.services.research_service.text.candidates.catalog.sets import (
    TEXT_CATALYST_V10_PROMOTED_SPECS,
    V10_ALPHA_QUALITY_TEXT_CANDIDATES,
    text_candidate_specs_by_name,
    text_candidates_for_set,
)
from quant_platform.services.research_service.text.candidates.catalog.types import (
    CandidateFormula,
    TextCandidateSpec,
)

__all__ = [
    "CandidateFormula",
    "TEXT_CATALYST_V10_PROMOTED_CANDIDATES",
    "TEXT_CATALYST_V10_PROMOTED_SPECS",
    "V10_ALPHA_QUALITY_TEXT_CANDIDATES",
    "TextCandidateSpec",
    "build_text_aggregate_features",
    "text_candidate_specs_by_name",
    "text_candidates_for_set",
]
