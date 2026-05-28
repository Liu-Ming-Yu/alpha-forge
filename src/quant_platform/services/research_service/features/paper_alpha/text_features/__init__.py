"""Paper-alpha text/catalyst feature materialization package."""

from __future__ import annotations

from quant_platform.services.research_service.features.paper_alpha.text_features.builders import (
    build_paper_alpha_catalyst_v10_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.text_features.versions import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_EVENT_FEATURE_SET_VERSION,
    TEXT_CATALYST_V3_EVENT_FEATURE_SET_VERSION,
    TEXT_CATALYST_V4_EVENT_FEATURE_SET_VERSION,
    TEXT_CATALYST_V5_EVENT_FEATURE_SET_VERSION,
    TEXT_EVENT_FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.paper_alpha.text_formulas import (
    TEXT_CATALYST_V10_ALPHA_FEATURES,
    TEXT_DECAY_LOOKBACK_DAYS,
)

__all__ = [
    "PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION",
    "TEXT_CATALYST_V10_ALPHA_FEATURES",
    "TEXT_DECAY_LOOKBACK_DAYS",
    "TEXT_CATALYST_EVENT_FEATURE_SET_VERSION",
    "TEXT_CATALYST_V3_EVENT_FEATURE_SET_VERSION",
    "TEXT_CATALYST_V4_EVENT_FEATURE_SET_VERSION",
    "TEXT_CATALYST_V5_EVENT_FEATURE_SET_VERSION",
    "TEXT_EVENT_FEATURE_SET_VERSION",
    "build_paper_alpha_catalyst_v10_feature_bundle",
]
