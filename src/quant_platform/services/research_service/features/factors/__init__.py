"""Classical alpha factor computations grouped by factor family."""

from __future__ import annotations

from quant_platform.services.research_service.features.factors.core import (
    InsufficientDataError,
    sma,
)
from quant_platform.services.research_service.features.factors.momentum import (
    momentum_return,
    momentum_skip1m,
    short_term_reversal,
    trend_quality,
)
from quant_platform.services.research_service.features.factors.price_levels import (
    distance_to_52w_high,
    mean_reversion,
    trend_z_score,
)
from quant_platform.services.research_service.features.factors.volatility import (
    low_volatility,
    realized_vol,
    vol_compression_ratio,
)

__all__ = [
    "InsufficientDataError",
    "distance_to_52w_high",
    "low_volatility",
    "mean_reversion",
    "momentum_return",
    "momentum_skip1m",
    "realized_vol",
    "short_term_reversal",
    "sma",
    "trend_quality",
    "trend_z_score",
    "vol_compression_ratio",
]
