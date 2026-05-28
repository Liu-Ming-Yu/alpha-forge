"""Current text and catalyst feature formula metadata."""

from __future__ import annotations

from quant_platform.services.research_service.text.candidates.catalog import (
    TEXT_CATALYST_V10_PROMOTED_CANDIDATES,
)

TEXT_DECAY_LOOKBACK_DAYS = 21
TEXT_CATALYST_V10_ALPHA_FEATURES = TEXT_CATALYST_V10_PROMOTED_CANDIDATES

__all__ = [
    "TEXT_CATALYST_V10_ALPHA_FEATURES",
    "TEXT_DECAY_LOOKBACK_DAYS",
]
