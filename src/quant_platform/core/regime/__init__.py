"""Stable facade for core market-regime helpers."""

from __future__ import annotations

from quant_platform.core.regime.classification import (
    RegimeStatsLike,
    RegimeThresholdsLike,
    classify_regime,
    detector_version,
    threshold_support,
)
from quant_platform.core.regime.detector import MarketRegimeDetector
from quant_platform.core.regime.states import (
    build_classified_state,
    build_no_stats_state,
    build_stable_state,
)
from quant_platform.core.regime.stats import MarketStats, compute_market_stats
from quant_platform.core.regime.thresholds import (
    DEFAULT_REGIME_THRESHOLDS,
    RegimeThresholds,
)

__all__ = [
    "DEFAULT_REGIME_THRESHOLDS",
    "MarketRegimeDetector",
    "MarketStats",
    "RegimeStatsLike",
    "RegimeThresholds",
    "RegimeThresholdsLike",
    "build_classified_state",
    "build_no_stats_state",
    "build_stable_state",
    "classify_regime",
    "compute_market_stats",
    "detector_version",
    "threshold_support",
]
