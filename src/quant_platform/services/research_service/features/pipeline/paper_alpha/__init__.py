"""Paper-alpha feature-set version constants.

Paper-alpha feature computation flows through the cross-section ``build_*``
builders and the shared ``compute_and_store_feature_bundle`` persistence
helper; this package only re-exports the canonical version identifiers.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.paper_alpha.composite import (
    PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.paper_alpha.event import (
    PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
)

__all__ = [
    "PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION",
    "PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION",
    "PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION",
]
