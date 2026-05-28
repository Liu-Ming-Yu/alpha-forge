"""Slippage calibration helpers for research campaigns."""

from __future__ import annotations


def calibrated_slippage_bps_per_turnover(
    observed_avg_slippage_bps: float | None,
    *,
    default_bps: float,
    floor_bps: float = 1.0,
    calibration_recommendation_bps: float | None = None,
) -> float:
    """Return the slippage assumption that should penalise OOS returns."""
    candidates = [default_bps, floor_bps]
    if observed_avg_slippage_bps is not None and observed_avg_slippage_bps > 0:
        candidates.append(float(observed_avg_slippage_bps))
    if calibration_recommendation_bps is not None and calibration_recommendation_bps > 0:
        candidates.append(float(calibration_recommendation_bps))
    return max(candidates)


__all__ = ["calibrated_slippage_bps_per_turnover"]
