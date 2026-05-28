"""Compatibility exports for pure market-regime classification logic."""

from __future__ import annotations

from quant_platform.core.regime import (
    RegimeStatsLike,
    RegimeThresholdsLike,
    classify_regime,
)

__all__ = ["RegimeStatsLike", "RegimeThresholdsLike", "classify_regime"]
