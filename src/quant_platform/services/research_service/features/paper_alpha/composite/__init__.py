"""Shared paper-alpha composite feature materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.features.cross_section.cross_section import (
    FeatureBundle,
    build_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.event import (
    build_paper_alpha_event_reaction_v2_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import FeatureVector

PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION = "paper-alpha-composite-v1"
PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION = "paper-alpha-event-reaction-v2"
PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION = (
    "paper-alpha-intraday-microstructure-v2"
)


async def build_paper_alpha_composite_feature_bundle(
    bar_data: Mapping[uuid.UUID, Sequence[MarketBar]],
    *,
    source_feature_repo: FeatureRepository,
    as_of: datetime,
    text_feature_set_version: str = PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    event_feature_set_version: str = PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
    intraday_feature_set_version: str = PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION,
) -> FeatureBundle:
    """Merge admitted paper source features into one runtime-compatible vector set."""
    close_history = {
        instrument_id: [float(bar.close) for bar in bars]
        for instrument_id, bars in bar_data.items()
    }
    classical = build_feature_bundle(close_history)
    event_bundle = build_paper_alpha_event_reaction_v2_feature_bundle(bar_data)
    instrument_ids = tuple(bar_data)
    source_vectors = await _load_source_vectors(
        source_feature_repo,
        instrument_ids=instrument_ids,
        as_of=as_of,
        feature_set_versions=(
            text_feature_set_version,
            event_feature_set_version,
            intraday_feature_set_version,
        ),
    )
    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for instrument_id in instrument_ids:
        row = dict(classical.alpha_features.get(instrument_id, {}))
        row.update(event_bundle.alpha_features.get(instrument_id, {}))
        for vector in source_vectors.get(instrument_id, ()):
            row.update(vector.features)
        alpha_features[instrument_id] = row
    return FeatureBundle(
        alpha_features=alpha_features,
        vol_forecasts=dict(classical.vol_forecasts),
    )


async def _load_source_vectors(
    repo: FeatureRepository,
    *,
    instrument_ids: Sequence[uuid.UUID],
    as_of: datetime,
    feature_set_versions: Sequence[str],
) -> dict[uuid.UUID, tuple[FeatureVector, ...]]:
    grouped: dict[uuid.UUID, list[FeatureVector]] = {
        instrument_id: [] for instrument_id in instrument_ids
    }
    for feature_set_version in feature_set_versions:
        vectors = await repo.get_vectors(list(instrument_ids), feature_set_version, as_of)
        for vector in vectors:
            grouped.setdefault(vector.instrument_id, []).append(vector)
    return {
        instrument_id: tuple(sorted(rows, key=lambda vector: vector.feature_set_version))
        for instrument_id, rows in grouped.items()
    }


__all__ = [
    "PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION",
    "PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION",
    "PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION",
    "build_paper_alpha_composite_feature_bundle",
]
