"""Pure market-regime classification logic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.domain.signals import RegimeLabel

if TYPE_CHECKING:
    from quant_platform.core.regime.thresholds import RegimeThresholds


class RegimeStatsLike(Protocol):
    @property
    def realized_vol(self) -> float: ...

    @property
    def trend_z(self) -> float: ...

    @property
    def breadth(self) -> float: ...


class RegimeThresholdsLike(Protocol):
    @property
    def hysteresis_vol(self) -> float: ...

    @property
    def crisis_vol(self) -> float: ...

    @property
    def risk_off_vol(self) -> float: ...

    @property
    def low_vol(self) -> float: ...

    @property
    def downtrend_z(self) -> float: ...

    @property
    def weak_breadth(self) -> float: ...

    @property
    def uptrend_z(self) -> float: ...

    @property
    def strong_breadth(self) -> float: ...


def classify_regime(
    stats: RegimeStatsLike,
    thresholds: RegimeThresholdsLike,
    current_label: RegimeLabel | None = None,
) -> tuple[RegimeLabel, float]:
    """Return (RegimeLabel, confidence) from market stats and thresholds."""
    v = stats.realized_vol
    z = stats.trend_z
    b = stats.breadth
    t = thresholds

    h = t.hysteresis_vol
    crisis_enter = t.crisis_vol
    crisis_exit = t.crisis_vol - h if current_label == RegimeLabel.CRISIS else t.crisis_vol
    risk_off_enter = t.risk_off_vol
    risk_off_exit = t.risk_off_vol - h if current_label == RegimeLabel.RISK_OFF else t.risk_off_vol
    low_vol_exit = t.low_vol + h if current_label == RegimeLabel.RISK_ON else t.low_vol

    if v >= crisis_enter:
        confidence = min(1.0, (v - t.crisis_vol) / 0.10 + 0.8)
        return RegimeLabel.CRISIS, round(min(confidence, 1.0), 4)
    if current_label == RegimeLabel.CRISIS and v >= crisis_exit:
        return RegimeLabel.CRISIS, 0.75
    if v >= risk_off_enter and z <= t.downtrend_z:
        return RegimeLabel.CRISIS, 0.75

    risk_off_signals = 0
    if v >= risk_off_exit:
        risk_off_signals += 1
    if z <= t.downtrend_z:
        risk_off_signals += 1
    if b <= t.weak_breadth:
        risk_off_signals += 1
    if risk_off_signals > 0:
        confidence = 0.5 + 0.15 * risk_off_signals
        return RegimeLabel.RISK_OFF, round(min(confidence, 1.0), 4)

    if v < low_vol_exit and z >= t.uptrend_z and b >= t.strong_breadth:
        vol_margin = (t.low_vol - v) / t.low_vol
        trend_margin = 1.0 if t.uptrend_z == 0.0 else z / t.uptrend_z
        denom_breadth = 1.0 - t.strong_breadth
        breadth_margin = 1.0 if denom_breadth <= 0.0 else (b - t.strong_breadth) / denom_breadth
        confidence = 0.6 + 0.4 * min(vol_margin, trend_margin, breadth_margin)
        return RegimeLabel.RISK_ON, round(min(confidence, 1.0), 4)

    return RegimeLabel.TRANSITION, 0.5


def detector_version(base_version: str, thresholds: RegimeThresholds) -> str:
    payload = json.dumps(asdict(thresholds), sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{base_version}-{digest}"


def threshold_support(thresholds: RegimeThresholds) -> dict[str, float]:
    return {
        "crisis_vol": thresholds.crisis_vol,
        "risk_off_vol": thresholds.risk_off_vol,
        "low_vol": thresholds.low_vol,
        "downtrend_z": thresholds.downtrend_z,
        "uptrend_z": thresholds.uptrend_z,
        "weak_breadth": thresholds.weak_breadth,
        "strong_breadth": thresholds.strong_breadth,
    }


__all__ = [
    "RegimeStatsLike",
    "RegimeThresholdsLike",
    "classify_regime",
    "detector_version",
    "threshold_support",
]
