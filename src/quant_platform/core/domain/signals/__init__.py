"""Signal and regime domain models.

SignalScore is the output of the signal service.  It carries a normalised
cross-sectional rank score and must not carry any position-sizing or
capital-allocation information.  That boundary is enforced by the portfolio
service consuming this type.

RegimeState is produced by the RegimeDetector and consumed by the
PortfolioConstructor to modulate position sizing and risk limits.

Invariants:
- SignalScore.score is in [-1.0, 1.0]; positive means long-favoured.
- SignalScore.confidence is in [0.0, 1.0].
- RegimeState.regime_label is a controlled vocabulary (see RegimeLabel).
- All timestamps are timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


class RegimeLabel(StrEnum):
    """Controlled vocabulary for market regime classifications."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITION = "transition"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalScore:
    """A single instrument's signal output from the signal model.

    The score is a normalised cross-sectional rank.  It carries no information
    about how much capital to allocate; that is the PortfolioConstructor's
    responsibility.

    Args:
        score_id: Stable system UUID.
        instrument_id: FK to Instrument.
        strategy_run_id: FK to StrategyRun that produced this score.
        as_of: UTC timestamp at which the score was computed.
        score: Normalised cross-sectional score in [-1.0, 1.0].
            Positive = favoured long; negative = disfavoured.
        confidence: Model confidence in [0.0, 1.0].  Values below a
            portfolio-service threshold may be excluded from target construction.
        model_version: Semantic version of the signal model.
        feature_vector_id: FK to FeatureVector used as input.

    Must never contain:
        Target weights, dollar amounts, share counts, or risk limits.
    """

    score_id: uuid.UUID
    instrument_id: uuid.UUID
    strategy_run_id: uuid.UUID
    as_of: datetime
    score: float
    confidence: float
    model_version: str
    feature_vector_id: uuid.UUID

    def __post_init__(self) -> None:
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [-1.0, 1.0], got {self.score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")


@dataclass(frozen=True)
class RegimeState:
    """A point-in-time market regime classification.

    Produced by the RegimeDetector and passed to PortfolioConstructor.
    The portfolio service uses this to select risk limits and position-sizing
    parameters appropriate to the current market environment.

    Args:
        regime_id: Stable system UUID.
        as_of: UTC timestamp of regime determination.
        regime_label: Controlled vocabulary label (see RegimeLabel).
        confidence: Detector's confidence in this classification [0.0, 1.0].
        detector_version: Semantic version of the regime detection model.
        supporting_features: Key metrics that drove the classification.
            Included for explainability and operator review; must not be used
            for downstream signal computation.
    """

    regime_id: uuid.UUID
    as_of: datetime
    regime_label: RegimeLabel
    confidence: float
    detector_version: str
    supporting_features: Mapping[str, object]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
