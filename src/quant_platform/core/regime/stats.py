"""Pure market-statistics inputs for regime classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.core.algorithms.price_factors import (
    InsufficientDataError,
    realized_vol,
    sma,
    trend_z_score,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


@dataclass(frozen=True)
class MarketStats:
    """Market-level statistics used as inputs to regime classification."""

    trend_z: float
    realized_vol: float
    breadth: float
    as_of: datetime

    def __post_init__(self) -> None:
        if not (0.0 <= self.breadth <= 1.0):
            raise ValueError(f"breadth must be in [0, 1], got {self.breadth}")
        if self.realized_vol < 0.0:
            raise ValueError(f"realized_vol must be non-negative, got {self.realized_vol}")
        if self.as_of.tzinfo is None:
            raise ValueError("MarketStats.as_of must be timezone-aware")


def compute_market_stats(
    index_closes: list[float],
    instrument_closes: dict[uuid.UUID, list[float]],
    as_of: datetime,
    trend_window: int = 200,
    vol_window: int = 21,
    breadth_window: int = 50,
) -> MarketStats:
    """Compute ``MarketStats`` from daily close-price series."""
    try:
        trend_z = trend_z_score(index_closes, trend_window)
    except InsufficientDataError:
        trend_z = 0.0

    try:
        vol = realized_vol(index_closes, vol_window, annualize=True)
    except InsufficientDataError:
        vol = 0.20

    above = 0
    total = 0
    for closes in instrument_closes.values():
        if len(closes) < breadth_window:
            continue
        try:
            moving_average = sma(closes, breadth_window)
            if closes[-1] > moving_average:
                above += 1
            total += 1
        except InsufficientDataError:
            pass

    breadth = above / total if total > 0 else 0.5

    return MarketStats(
        trend_z=trend_z,
        realized_vol=vol,
        breadth=breadth,
        as_of=as_of,
    )


__all__ = ["MarketStats", "compute_market_stats"]
