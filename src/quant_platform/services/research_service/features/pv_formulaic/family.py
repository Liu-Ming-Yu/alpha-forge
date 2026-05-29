"""Live ``pv_formulaic`` feature family — FeatureBundle assembly (ADR-011 increment 2b).

Turns live ``MarketBar`` payloads into a :class:`FeatureBundle` of the pv+formulaic
features (latest row per instrument), via the bars adapter + the kernel compute.

Transform note (the parity decision): this family carries **raw** kernel-computed
feature values, *unlike* the ``close`` family which rank-normalises into ``[-1,1]``.
That is deliberate and required for fidelity to Arm G:

* ``build_supervised_samples`` feeds the research ranker **raw** feature values
  (no per-date normalisation), and ``score_features`` is a **raw weighted sum**
  ``Σ feature·ic_weight``.
* So a raw bundle + G's promoted IC weights, scored by ``LinearWeightSignalModel``
  (which applies its own rank-preserving ``[-1,1]`` normalisation to the
  *weighted-sum score*, not the inputs), reproduces G's cross-sectional ranking —
  and therefore the long-only top-N selection — by construction.

Exact holding sizes are reconciled against the G backtest in the simulated-backend
validation (ADR-011 increment 4) before anything trades.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_platform.services.research_service.features.cross_section.cross_section import (
    FeatureBundle,
)
from quant_platform.services.research_service.features.pv_formulaic.bars_frame import (
    market_bars_to_ohlcv_frame,
)
from quant_platform.services.research_service.features.pv_formulaic.compute import (
    PV_FORMULAIC_FEATURE_NAMES,
    compute_pv_formulaic_frame,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.domain.market_data.bars import MarketBar

#: Feature-set version pinned onto the live pv_formulaic family / its evidence.
PV_FORMULAIC_FEATURE_SET_VERSION = "pv-formulaic-live-v1"


def build_pv_formulaic_feature_bundle(
    bars_by_instrument: Mapping[uuid.UUID, Sequence[MarketBar]],
    *,
    as_of: datetime | None = None,
) -> FeatureBundle:
    """Build a raw-valued pv+formulaic :class:`FeatureBundle` for the latest bar.

    For each instrument, the most recent ``as_of``-eligible row of the
    pv+formulaic feature matrix becomes ``{feature_name: raw_value}``. Non-finite
    features are omitted; an instrument with no finite features is absent from the
    bundle (matching the ``close`` family's behaviour). Returns an empty bundle
    when there are no in-window bars.
    """
    frame = market_bars_to_ohlcv_frame(bars_by_instrument, as_of=as_of)
    if frame.empty:
        return FeatureBundle()
    combined = compute_pv_formulaic_frame(frame)
    latest = combined.sort_values("date").groupby("instrument_id", sort=False).tail(1)
    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for _, row in latest.iterrows():
        feats = {
            name: float(row[name])
            for name in PV_FORMULAIC_FEATURE_NAMES
            if name in row.index and pd.notna(row[name])
        }
        if feats:
            alpha_features[row["instrument_id"]] = feats
    return FeatureBundle(alpha_features=alpha_features)


__all__ = ["PV_FORMULAIC_FEATURE_SET_VERSION", "build_pv_formulaic_feature_bundle"]
