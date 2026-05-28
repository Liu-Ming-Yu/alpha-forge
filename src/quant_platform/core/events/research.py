"""Research and signal domain events."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from quant_platform.core.domain.signals import RegimeLabel
from quant_platform.core.events.base import DomainEvent


@dataclass(frozen=True)
class FeatureVectorComputed(DomainEvent):
    """A feature vector is ready for signal scoring.

    Args:
        vector_id: FK to FeatureVector.
        instrument_id: Instrument the vector belongs to.
        strategy_run_id: FK to the StrategyRun that requested computation.
    """

    vector_id: uuid.UUID
    instrument_id: uuid.UUID
    strategy_run_id: uuid.UUID


@dataclass(frozen=True)
class SignalScorePublished(DomainEvent):
    """A signal score has been computed and is ready for portfolio construction.

    Args:
        score_id: FK to SignalScore.
        instrument_id: Scored instrument.
        strategy_run_id: FK to the StrategyRun.
    """

    score_id: uuid.UUID
    instrument_id: uuid.UUID
    strategy_run_id: uuid.UUID


# ---------------------------------------------------------------------------
# Research service events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestCompleted(DomainEvent):
    """A backtest simulation run has completed successfully.

    Args:
        backtest_id: FK to the BacktestRun summary record.
        strategy_run_id: FK to the StrategyRun that drove the simulation.
        artifact_uri: URI to the Parquet file containing per-trade results.
    """


@dataclass(frozen=True)
class RegimeStateDetected(DomainEvent):
    """A market regime classification has been computed for a cycle.

    Emitted after ``RegimeDetector.detect()`` so downstream read models can
    surface the latest regime without having to poll the detector directly.

    Args:
        regime_id: FK to :class:`RegimeState`.
        regime_label: Controlled vocabulary label (risk_on/risk_off/...).
        confidence: Detector confidence in ``[0.0, 1.0]``.
        gross_exposure_scale: Resolved gross-exposure multiplier for this regime
            (e.g. 0.0 on CRISIS, 1.0 on RISK_ON).
        supporting_features: Key metrics that drove the classification
            (``trend_z``, ``annualized_vol``, ``breadth_pct``, ...).
    """

    regime_id: uuid.UUID
    regime_label: RegimeLabel
    confidence: float
    gross_exposure_scale: float
    supporting_features: Mapping[str, object]
