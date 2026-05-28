"""Public builders for governed paper-alpha text feature bundles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.features.paper_alpha.text_features.internals import (
    _build_text_candidate_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.text_features.versions import (
    TEXT_CATALYST_V4_EVENT_FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.paper_alpha.text_formulas import (
    TEXT_CATALYST_V10_ALPHA_FEATURES,
    TEXT_DECAY_LOOKBACK_DAYS,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.features.cross_section.cross_section import (
        FeatureBundle,
    )


async def build_paper_alpha_catalyst_v10_feature_bundle(
    bar_data: Mapping[uuid.UUID, Sequence[MarketBar]],
    *,
    text_feature_repo: FeatureRepository,
    as_of: datetime,
    text_feature_set_version: str = TEXT_CATALYST_V4_EVENT_FEATURE_SET_VERSION,
    lookback_days: int = TEXT_DECAY_LOOKBACK_DAYS,
) -> FeatureBundle:
    """Build current alpha-quality primary SEC filing features."""
    return await _build_text_candidate_feature_bundle(
        bar_data=bar_data,
        text_feature_repo=text_feature_repo,
        as_of=as_of,
        text_feature_set_version=text_feature_set_version,
        lookback_days=lookback_days,
        feature_names=TEXT_CATALYST_V10_ALPHA_FEATURES,
        candidate_set="v10-alpha-quality",
    )
