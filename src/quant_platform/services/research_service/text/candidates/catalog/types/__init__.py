"""Shared text-candidate types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

CandidateFormula = Callable[[Mapping[str, float], Mapping[str, float], float], float]


@dataclass(frozen=True)
class TextCandidateSpec:
    """One prospective formula for read-only text candidate screening."""

    name: str
    formula: CandidateFormula
    expression: str
    thesis: str = ""
