"""Shared intraday-candidate types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import (
    CandidateScreenThresholds,
)

if TYPE_CHECKING:
    from quant_platform.services.research_service.intraday.candidates.features import (
        IntradayFormula,
    )

IntradayCandidateScreenThresholds = CandidateScreenThresholds
INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION = "paper-alpha-intraday-microstructure-v1"
INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION = "paper-alpha-intraday-microstructure-v2"


@dataclass(frozen=True)
class IntradayCandidateSpec:
    """One prospective 1-minute bar-structure formula for diagnostic screening."""

    name: str
    formula: IntradayFormula
    expression: str
    thesis: str
    lookback_days: int
