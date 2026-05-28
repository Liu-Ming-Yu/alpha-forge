"""Compatibility exports for market-regime detection."""

from __future__ import annotations

from quant_platform.core.regime import (
    MarketRegimeDetector,
    MarketStats,
    RegimeThresholds,
)
from quant_platform.core.regime import (
    classify_regime as _classify,
)

__all__ = ["MarketRegimeDetector", "MarketStats", "RegimeThresholds", "_classify"]
