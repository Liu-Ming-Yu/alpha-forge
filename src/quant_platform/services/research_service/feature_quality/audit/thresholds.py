"""Threshold DTOs for institutional feature audits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureAuditThresholds:
    """Conservative defaults for feature promotion evidence."""

    min_daily_groups: int = 252
    min_coverage: float = 0.95
    min_dispersion: float = 0.01
    min_unique_ratio: float = 0.05
    max_zero_fraction: float = 0.95
    max_abs_ic: float = 0.99
    min_lag_ic_fraction: float = 0.10
    min_oos_ic: float = 0.02
    min_icir: float = 0.10
    max_negative_ic_streak: int = 3
    max_turnover: float = 4.0
    min_net_mean_return: float = 0.0
    min_incremental_ic_delta: float = 0.0
    max_baseline_correlation: float = 0.95
    require_rank_normalized: bool = True
    """Block alpha features that are not cross-sectionally rank-normalized.

    A linear ensemble sums ``value * weight`` across factors; mixing raw-scale
    factors (returns vs. volatility ratios) silently lets one unit dominate.
    Enforcing rank-normalization to [-1, 1] is what keeps the blend coherent.
    """
