"""DTOs and constants for factor-weight calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

MOMENTUM_BLOC: tuple[str, ...] = (
    "momentum_1m",
    "momentum_3m",
    "momentum_12m_1m",
    "vol_compression",
)

ALPHA_BLOC: tuple[str, ...] = (
    "short_term_reversal_5d",
    "trend_quality_63d",
    "distance_to_52w_high",
)


@dataclass(frozen=True)
class CalibrationSample:
    """One feature-vector and forward-return pair used for fitting."""

    as_of: datetime
    instrument_id: uuid.UUID
    features: dict[str, float]
    forward_return: float


@dataclass(frozen=True)
class CalibratedWeights:
    """Output of a calibration run, written as a JSON artifact."""

    as_of: datetime
    weights: dict[str, float]
    sample_size: int
    r_squared_momentum: float
    r_squared_alpha: float
    l2_lambda: float
    horizon_days: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["as_of"] = self.as_of.astimezone(UTC).isoformat()
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> CalibratedWeights:
        data = json.loads(raw)
        data["as_of"] = datetime.fromisoformat(data["as_of"])
        return cls(**data)


__all__ = [
    "ALPHA_BLOC",
    "MOMENTUM_BLOC",
    "CalibratedWeights",
    "CalibrationSample",
]
