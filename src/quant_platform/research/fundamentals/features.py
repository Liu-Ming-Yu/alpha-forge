"""Legacy 9-feature Sharadar starter — compatibility shim.

The original 9 quality+value features that the campaign evaluator
admitted in 2026-05 (see :mod:`project_fundamentals_starter_verdict`)
now live as a subset of the larger
:mod:`quant_platform.research.features.fundamentals` family
(``fundamentals-plus-v1``). This module exists only so the existing
walk-forward script, evidence bundles, and unit tests that import
``compute_starter_features`` / ``FEATURE_NAMES`` / ``EXPECTED_SIGNS``
/ ``FeatureFrame`` from ``quant_platform.research.fundamentals.features``
keep working without churn.

The compat surface preserved here:

* ``FEATURE_NAMES`` — the 9-name tuple, in its original order.
* ``EXPECTED_SIGNS`` — ``{name: +1}`` for every name, used by the
  non-negative-weight classical fitter.
* ``FeatureFrame`` — the legacy dataclass (``frame``, ``feature_names``,
  ``expected_signs``, ``coverage``). Distinct from the new
  :class:`quant_platform.research.features.contracts.FeatureFrame`
  (which carries ``feature_specs`` and ``key_columns``); both keep
  living alongside each other for now.
* ``compute_starter_features(panel, *, sector_neutralize, sector_map)``
  — same signature as before, internally delegates to
  ``compute_fundamentals_features`` and filters down to the legacy 9
  columns.

New code should import from
``quant_platform.research.features.fundamentals`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.research.features.fundamentals.features import (
    compute_fundamentals_features,
)
from quant_platform.research.features.neutralization import neutralize_feature_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from quant_platform.research.fundamentals.sharadar import SharadarPanel


#: Ordered tuple of the legacy 9 feature names. Pinned here so any
#: caller that imports ``FEATURE_NAMES`` from this module sees exactly
#: the historical 9-feature surface, even after
#: ``fundamentals-plus-v1`` grows.
FEATURE_NAMES: tuple[str, ...] = (
    "roe_ttm",
    "gross_profitability_q",
    "low_accruals_4q",
    "low_asset_growth_yoy",
    "fcf_yield_ttm",
    "cash_to_assets",
    "low_debt_to_equity",
    "book_to_price",
    "earnings_to_price",
)

#: ``{name: +1}`` for every legacy feature — the non-negative-weight
#: evaluator depends on every classical fitter input being
#: positive-oriented. Re-derived from :data:`FEATURE_NAMES` so the two
#: cannot drift.
EXPECTED_SIGNS: Mapping[str, int] = {name: +1 for name in FEATURE_NAMES}


@dataclass(frozen=True)
class FeatureFrame:
    """Result wrapper preserved for callers that depend on the legacy shape.

    Distinct from
    :class:`quant_platform.research.features.contracts.FeatureFrame`,
    which carries ``feature_specs`` and ``key_columns``; downstream
    walk-forward code reads ``expected_signs`` directly off this
    dataclass so we keep both shapes alive.
    """

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    expected_signs: Mapping[str, int]
    coverage: Mapping[str, int]


def compute_starter_features(
    panel: SharadarPanel,
    *,
    sector_neutralize: bool = False,
    sector_map: Mapping[str, str] | None = None,
) -> FeatureFrame:
    """Compute the legacy 9 quality+value features.

    Internally this is now a thin filter over
    :func:`quant_platform.research.features.fundamentals.compute_fundamentals_features`:
    the new compute produces ~40 features, we keep the 9 the legacy
    catalog admitted, and we wrap the result in the legacy
    :class:`FeatureFrame` shape so existing callers do not change.

    Parameters
    ----------
    panel:
        Loaded :class:`SharadarPanel`.
    sector_neutralize:
        When ``True``, every feature is replaced by
        ``feature - sector_median`` per ``(datekey, sector)``. Delegates
        to the shared
        :func:`quant_platform.research.features.neutralization.neutralize_by_group`
        helper.
    sector_map:
        ``instrument_id -> sector`` mapping required when
        ``sector_neutralize=True``.
    """
    if sector_neutralize and sector_map is None:
        raise ValueError("compute_starter_features: sector_neutralize=True requires sector_map")

    plus_frame = compute_fundamentals_features(panel)
    if sector_neutralize:
        plus_frame = neutralize_feature_frame(
            plus_frame,
            by="sector_median",
            sector_map=sector_map,
        )

    # Filter to the legacy 9 names and the carry-through columns the
    # walk-forward script expects.
    keep_cols = ["instrument_id", "datekey"]
    for col in ("ticker", "calendardate"):
        if col in plus_frame.frame.columns:
            keep_cols.append(col)
    keep_cols.extend(FEATURE_NAMES)

    legacy_frame = plus_frame.frame.loc[:, keep_cols].copy().reset_index(drop=True)
    coverage = {name: int(legacy_frame[name].notna().sum()) for name in FEATURE_NAMES}

    return FeatureFrame(
        frame=legacy_frame,
        feature_names=FEATURE_NAMES,
        expected_signs=EXPECTED_SIGNS,
        coverage=coverage,
    )


__all__ = [
    "EXPECTED_SIGNS",
    "FEATURE_NAMES",
    "FeatureFrame",
    "compute_starter_features",
]
