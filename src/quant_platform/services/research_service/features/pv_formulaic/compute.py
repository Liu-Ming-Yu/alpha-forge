"""Live pv+formulaic feature compute via the inner-layer kernel (ADR-011 increment 2).

Reproduces the research backtest's pv+formulaic feature matrix
(``price_volume.compute_price_volume_features`` + the formulaic ``LIBRARY``
evaluated over a ``MarketPanel``) on live bars, reusing the *same* kernel compute
the research factory uses — so the values match the research computation by
construction (same code, not a re-implementation).

This module is the pure feature matrix. The ``FeatureBundle`` assembly +
normalization contract + family registration, and the reconciliation of the
scoring transform with G's research ranker, are the next sub-steps (increment 2b
/ the G plugin), validated end-to-end in the simulated run.
"""

from __future__ import annotations

import pandas as pd

from quant_platform.services.research_service.features.kernel.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.services.research_service.features.kernel.formulaic.library import LIBRARY
from quant_platform.services.research_service.features.kernel.formulaic.panel import (
    build_market_panel,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    FEATURE_NAMES as PV_FEATURE_NAMES,
)
from quant_platform.services.research_service.features.kernel.price_volume.features import (
    compute_price_volume_features,
)

#: Formulaic alpha names from the curated LIBRARY (includes G's wq alphas).
FORMULAIC_FEATURE_NAMES: tuple[str, ...] = tuple(alpha.name for alpha in LIBRARY)

#: Full pv+formulaic feature surface this family produces (price-volume + formulaic).
PV_FORMULAIC_FEATURE_NAMES: tuple[str, ...] = tuple(PV_FEATURE_NAMES) + FORMULAIC_FEATURE_NAMES


def compute_pv_formulaic_frame(bars_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the ``(instrument_id, date)``-grained pv+formulaic feature matrix.

    Mirrors the research backtest's assembly: price-volume features left-merged
    with the formulaic ``LIBRARY`` evaluated over a ``MarketPanel``, both from the
    shared kernel compute. Every price-volume row keeps its formulaic columns.
    """
    pv = compute_price_volume_features(bars_frame).frame
    panel = build_market_panel(bars_frame)
    cache = ExpressionCache()
    formulaic = pd.DataFrame(
        {
            "instrument_id": panel.frame["instrument_id"].to_numpy(),
            "date": panel.frame["date"].to_numpy(),
        }
    )
    for alpha in LIBRARY:
        formulaic[alpha.name] = (
            evaluate_expression(panel, alpha.expression, cache=cache).astype(float).to_numpy()
        )
    return pv.merge(formulaic, on=["instrument_id", "date"], how="left")


__all__ = [
    "FORMULAIC_FEATURE_NAMES",
    "PV_FORMULAIC_FEATURE_NAMES",
    "compute_pv_formulaic_frame",
]
