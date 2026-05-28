"""Feature-set version constants and shared feature-pipeline helpers.

Feature computation now flows through ``FeatureFamilyRegistry`` plugins
(``services/research_service/features/plugins.py``); this module only owns the
canonical ``*_FEATURE_SET_VERSION`` identifiers and the reserved vol-forecast
helpers that both the registry path and downstream consumers share.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.pipeline.paper_alpha import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
    PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.pipeline.storage import (
    VOL_FORECAST_KEY as VOL_FORECAST_KEY,
)
from quant_platform.services.research_service.features.pipeline.storage import (
    extract_vol_forecasts as extract_vol_forecasts,
)

# 1.1.0 adds the WS4 diversifying factors (reversal_21d, low_volatility_63d,
# mean_reversion_63d) to the close family.  Bumped from 1.0.0 so stale feature
# data missing the new factors fails closed instead of scoring silently.
FEATURE_SET_VERSION = "1.1.0"


__all__ = [
    "FEATURE_SET_VERSION",
    "PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION",
    "PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION",
    "PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION",
    "VOL_FORECAST_KEY",
    "extract_vol_forecasts",
]
