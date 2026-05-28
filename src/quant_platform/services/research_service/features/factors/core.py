"""Compatibility exports for shared pure price-series factor primitives."""

from __future__ import annotations

from quant_platform.core.algorithms.price_factors import (
    InsufficientDataError,
    sma,
    trailing_log_returns,
)

__all__ = ["InsufficientDataError", "sma", "trailing_log_returns"]
