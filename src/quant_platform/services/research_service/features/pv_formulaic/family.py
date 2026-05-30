"""Live ``pv_formulaic`` feature family — FeatureBundle assembly (ADR-011).

Turns live ``MarketBar`` payloads into a :class:`FeatureBundle` of the pv+formulaic
features (latest row per instrument), via the bars adapter + the kernel compute.

Transform note (the parity decision — CORRECTED). This family **rank-normalizes**
each feature cross-sectionally before emitting, using the same kernel
(:func:`cross_sectional_rank_normalize`) the research ranker applies.

History: increment-2b originally emitted **raw** values, on the (wrong) reasoning
that a raw bundle + G's IC weights + ``LinearWeightSignalModel`` reproduced the
backtest "by construction". The increment-4 parity check disproved that — a raw
``Σ feature·weight`` is dominated by ``dollar_volume_20d``'s ~1e8 scale (~1e8:1)
and the score model's ``[-1,1]`` clamp collapsed every name to a tie (live top-30
overlapped the backtest only 3/30). The dollar-volume fix (ADR-011) rank-normalizes
in the research ranker; this family does the **same** normalization so live scoring
equals backtest scoring — verified end-to-end by ``scripts/validate_arm_q_live_parity.py``
(0 clamped, 30/30 overlap, weight L1 0.0). See :func:`build_pv_formulaic_feature_bundle`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_platform.services.research_service.features.cross_section.cross_section import (
    FeatureBundle,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    cross_sectional_rank_normalize,
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
    """Build a rank-normalized pv+formulaic :class:`FeatureBundle` for the latest bar.

    For each instrument, the most recent ``as_of``-eligible row of the
    pv+formulaic feature matrix becomes ``{feature_name: value}``, where each
    feature is **cross-sectionally rank-normalized** across the as-of universe.

    The rank-normalization is the live half of the dollar-volume scoring fix
    (ADR-011): the research ranker rank-normalizes each feature within its
    cross-section before the weighted sum, because a *raw* ``Σ feature·weight`` is
    dominated by ``dollar_volume_20d``'s ~1e8 scale (~1e8:1) and collapses under
    the score model's [-1,1] clamp. Emitting raw values here (the original
    increment-2b shape) made the live top-N an arbitrary tie — the increment-4
    parity defect (3/30 overlap). Normalizing here with the **same kernel**
    (:func:`cross_sectional_rank_normalize`) the backtest uses makes the live
    cross-sectional ranking — and thus the top-N selection and the
    conviction-weighted book — match the backtest by construction.

    A name with no finite features is absent from the bundle; the normalizer fills
    NaN ranks with the neutral median (0.5). Returns an empty bundle when there
    are no in-window bars.
    """
    frame = market_bars_to_ohlcv_frame(bars_by_instrument, as_of=as_of)
    if frame.empty:
        return FeatureBundle()
    combined = compute_pv_formulaic_frame(frame)
    latest = combined.sort_values("date").groupby("instrument_id", sort=False).tail(1)
    feature_cols = [name for name in PV_FORMULAIC_FEATURE_NAMES if name in latest.columns]
    # Rank-normalize per date (the latest rows share the as-of date) so the live
    # weighted-sum score equals the backtest's rank-normalized ranker score.
    normed = cross_sectional_rank_normalize(latest, feature_cols, date_column="date")
    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for _, row in normed.iterrows():
        feats = {name: float(row[name]) for name in feature_cols if pd.notna(row[name])}
        if feats:
            alpha_features[row["instrument_id"]] = feats
    return FeatureBundle(alpha_features=alpha_features)


__all__ = ["PV_FORMULAIC_FEATURE_SET_VERSION", "build_pv_formulaic_feature_bundle"]
