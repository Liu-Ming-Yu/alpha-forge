"""Threshold contracts for market-regime classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeThresholds:
    """Regime classification thresholds."""

    crisis_vol: float = 0.35
    risk_off_vol: float = 0.25
    low_vol: float = 0.20
    downtrend_z: float = -0.05
    uptrend_z: float = 0.02
    weak_breadth: float = 0.40
    strong_breadth: float = 0.55
    hysteresis_vol: float = 0.005
    stability_window: int = 3


DEFAULT_REGIME_THRESHOLDS = RegimeThresholds()


__all__ = ["DEFAULT_REGIME_THRESHOLDS", "RegimeThresholds"]
