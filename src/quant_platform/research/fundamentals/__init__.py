"""Research-only fundamentals surface.

This package wraps vendor fundamentals data (Sharadar SF1 today) into a tidy
point-in-time panel suitable for walk-forward IC analysis. It deliberately
**does not** go through ``FeatureFamilyRegistry``; that production path is
reserved for feature sets that have already passed the eligibility gate.

The contract is:

* Input — parquet cache produced by ``scripts/pull_sharadar_sf1.py`` plus the
  ticker map produced by ``scripts/build_sharadar_ticker_map.py``.
* Output — a long-format ``pandas.DataFrame`` keyed by
  ``(instrument_id, datekey)``, with raw SF1 columns preserved and any
  derived columns documented per call.

Promote to ``services/research_service/features/fundamentals/`` only after a
walk-forward run produces a passing ``eligibility.json``.
"""

from __future__ import annotations

from quant_platform.research.fundamentals.features import (
    EXPECTED_SIGNS,
    FEATURE_NAMES,
    FeatureFrame,
    compute_starter_features,
)
from quant_platform.research.fundamentals.sharadar import (
    SharadarPanel,
    load_sector_map,
    load_sharadar_sf1_panel,
)

__all__ = [
    "EXPECTED_SIGNS",
    "FEATURE_NAMES",
    "FeatureFrame",
    "SharadarPanel",
    "compute_starter_features",
    "load_sector_map",
    "load_sharadar_sf1_panel",
]
