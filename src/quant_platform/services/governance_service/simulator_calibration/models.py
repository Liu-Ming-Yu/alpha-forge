"""Typed simulator calibration DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class FillObservation:
    """One paper-fill row used as calibration input."""

    tactic: str
    side: str
    quantity: int
    adv_shares_20d: float
    spread_bps: float
    slippage_bps: float
    executed_at: datetime
    stale_price_age_seconds: float = 0.0
    order_type: str = "unknown"


@dataclass(frozen=True)
class CalibrationBucket:
    """Recommended slippage for one execution-realism calibration cell."""

    tactic: str
    adv_bucket: str
    sample_count: int
    median_slippage_bps: float
    p90_slippage_bps: float
    recommended_bps: float
    spread_bucket: str = "all"
    stale_price_bucket: str = "all"
    order_type: str = "all"
    time_bucket: str = "all"


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregate calibration recommendations."""

    generated_at: datetime
    sample_count: int
    overall_median_bps: float
    overall_p90_bps: float
    overall_recommended_bps: float
    buckets: tuple[CalibrationBucket, ...]
    insufficient_data: bool
    floor_bps: float
    extras: dict[str, object] = field(default_factory=dict)
