"""Deterministic current candidate catalog for text-derived alpha screens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.text.candidates.catalog.fields import (
    TEXT_CATALYST_V10_PROMOTED_CANDIDATES,
)
from quant_platform.services.research_service.text.candidates.catalog.sets_current import (
    v10_alpha_quality_text_candidates,
)

if TYPE_CHECKING:
    from quant_platform.services.research_service.text.candidates.catalog.types import (
        TextCandidateSpec,
    )


def text_candidates_for_set(candidate_set: str) -> tuple[TextCandidateSpec, ...]:
    """Return the deterministic candidate catalog for a screen mode."""
    normalized = candidate_set.strip().lower()
    if normalized in {"v10-alpha-quality", "v10_alpha_quality"}:
        return V10_ALPHA_QUALITY_TEXT_CANDIDATES
    raise ValueError(f"unknown text candidate set: {candidate_set}")


def text_candidate_specs_by_name(
    candidate_set: str = "v10-alpha-quality",
) -> dict[str, TextCandidateSpec]:
    """Return candidate specs indexed by name for materialization/promotion."""
    return {candidate.name: candidate for candidate in text_candidates_for_set(candidate_set)}


V10_ALPHA_QUALITY_TEXT_CANDIDATES: tuple[TextCandidateSpec, ...] = (
    v10_alpha_quality_text_candidates()
)
TEXT_CATALYST_V10_PROMOTED_SPECS: tuple[TextCandidateSpec, ...] = tuple(
    text_candidate_specs_by_name("v10-alpha-quality")[name]
    for name in TEXT_CATALYST_V10_PROMOTED_CANDIDATES
)


__all__ = [
    "TEXT_CATALYST_V10_PROMOTED_SPECS",
    "V10_ALPHA_QUALITY_TEXT_CANDIDATES",
    "text_candidate_specs_by_name",
    "text_candidates_for_set",
]
