"""Types for intraday candidate feature derivation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime


IntradayFormula = Callable[["IntradayFeatureContext"], float]
INTRADAY_ALPHA_SCALE = 10_000.0


class IntradayCandidateFeatureSpec(Protocol):
    name: str
    formula: IntradayFormula
    lookback_days: int


@dataclass(frozen=True)
class IntradaySessionMetrics:
    instrument_id: uuid.UUID
    session_date: date
    start_at: datetime
    end_at: datetime
    opening_drive: float
    close_pressure: float
    vwap_pressure: float
    intraday_volatility: float
    range_expansion: float
    session_return: float
    volume_share: float


@dataclass(frozen=True)
class IntradayFeatureContext:
    metrics: IntradaySessionMetrics | None
    decay: float
    history: tuple[IntradaySessionMetrics, ...] = ()
    as_of: datetime | None = None
    sample_features: Mapping[str, float] | None = None


@dataclass(frozen=True)
class IntradayCandidateFeatureRow:
    as_of: datetime
    instrument_id: uuid.UUID
    features: Mapping[str, float]
    available_at: datetime
