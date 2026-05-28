"""Regime-state builders shared by services and bootstrap wiring."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.regime.classification import threshold_support

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.regime.stats import MarketStats
    from quant_platform.core.regime.thresholds import RegimeThresholds


def build_classified_state(
    *,
    stats: MarketStats,
    label: RegimeLabel,
    confidence: float,
    version: str,
    thresholds: RegimeThresholds,
) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=stats.as_of,
        regime_label=label,
        confidence=confidence,
        detector_version=version,
        supporting_features={
            "trend_z": stats.trend_z,
            "realized_vol": stats.realized_vol,
            "breadth": stats.breadth,
            "thresholds": threshold_support(thresholds),
        },
    )


def build_stable_state(
    *,
    stats: MarketStats,
    label: RegimeLabel,
    candidate_label: RegimeLabel,
    confidence: float,
    version: str,
    stability_window: int,
    disagree_haircut: float = 0.75,
) -> RegimeState:
    """Construct a stabilised RegimeState."""
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=stats.as_of,
        regime_label=label,
        confidence=(confidence if label == candidate_label else confidence * disagree_haircut),
        detector_version=version,
        supporting_features={
            "trend_z": stats.trend_z,
            "realized_vol": stats.realized_vol,
            "breadth": stats.breadth,
            "candidate_label": candidate_label.value,
            "stable_label": label.value,
            "stability_window": stability_window,
        },
    )


def build_no_stats_state(*, as_of: datetime, version: str) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=as_of,
        regime_label=RegimeLabel.TRANSITION,
        confidence=0.0,
        detector_version=version,
        supporting_features={"warning": "no MarketStats loaded via update()"},
    )


__all__ = [
    "build_classified_state",
    "build_no_stats_state",
    "build_stable_state",
]
