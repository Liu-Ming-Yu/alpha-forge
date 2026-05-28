"""Typed IC report DTOs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping


@dataclass(frozen=True)
class ICPanel:
    """One rebalance's cross-sectional snapshot of features + forward returns."""

    as_of: datetime
    features: Mapping[uuid.UUID, Mapping[str, float]]
    forward_returns: Mapping[uuid.UUID, float]


@dataclass(frozen=True)
class ICSeries:
    """Per-factor IC time series plus rolling stats and decay entries."""

    factor: str
    timestamps: tuple[datetime, ...]
    ic: tuple[float, ...]
    rolling_mean_20: tuple[float, ...]
    rolling_std_20: tuple[float, ...]
    decay: Mapping[int, float]

    def latest(self) -> float | None:
        for value in reversed(self.ic):
            if not np.isnan(value):
                return float(value)
        return None


@dataclass(frozen=True)
class ICReport:
    """All-factor bundle written to ``data/backtest/<run_id>/ic_report.json``."""

    run_id: uuid.UUID
    as_of: datetime
    horizons: tuple[int, ...]
    series: tuple[ICSeries, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "run_id": str(self.run_id),
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "horizons": list(self.horizons),
            "series": [
                {
                    "factor": s.factor,
                    "timestamps": [ts.astimezone(UTC).isoformat() for ts in s.timestamps],
                    "ic": [_finite_or_none(x) for x in s.ic],
                    "rolling_mean_20": [_finite_or_none(x) for x in s.rolling_mean_20],
                    "rolling_std_20": [_finite_or_none(x) for x in s.rolling_std_20],
                    "decay": {str(k): v for k, v in s.decay.items()},
                }
                for s in self.series
            ],
            "metadata": self.metadata,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _finite_or_none(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)
