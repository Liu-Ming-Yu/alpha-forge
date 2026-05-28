"""Rolling IC tracking for text-shadow signals."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

import structlog

from quant_platform.services.research_service.reports.statistics import spearman_ic

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

log = structlog.get_logger(__name__)


class ShadowDayRecord(NamedTuple):
    """One day's shadow scores indexed by instrument id."""

    as_of: datetime
    scores: dict[uuid.UUID, float]
    prices: dict[uuid.UUID, Decimal]


def shadow_spearman_ic(predicted: list[float], realized: list[float]) -> float:
    """Spearman IC with NaN semantics used by the shadow scorer."""
    return spearman_ic(
        predicted,
        realized,
        invalid_value=float("nan"),
        constant_value=float("nan"),
    )


class ShadowICTracker:
    """Maintains score history and rolling IC for shadow text signals."""

    def __init__(self, *, ic_window: int) -> None:
        self.score_history: deque[ShadowDayRecord] = deque(maxlen=ic_window + 1)
        self.ic_history: deque[float] = deque(maxlen=ic_window)

    @property
    def rolling_ic(self) -> float:
        valid = [value for value in self.ic_history if not math.isnan(value)]
        if len(valid) < 2:
            return float("nan")
        return sum(valid) / len(valid)

    @property
    def observations(self) -> int:
        return len(self.ic_history)

    def record_scores(
        self,
        *,
        as_of: datetime,
        scores: dict[uuid.UUID, float],
        market_prices: dict[uuid.UUID, Decimal] | None,
    ) -> None:
        if not scores:
            return
        cycle_prices = {
            instrument_id: market_prices[instrument_id]
            for instrument_id in scores
            if market_prices is not None and instrument_id in market_prices
        }
        self.score_history.append(ShadowDayRecord(as_of=as_of, scores=scores, prices=cycle_prices))

    def update_ic(self, market_prices: dict[uuid.UUID, Decimal]) -> None:
        """Compute IC between the latest scores and current prices."""
        if len(self.score_history) < 1:
            return

        prev_record = self.score_history[-1]
        common = set(prev_record.scores) & set(prev_record.prices) & set(market_prices)
        if len(common) < 2:
            return

        predicted: list[float] = []
        realized: list[float] = []
        for instrument_id in sorted(common):
            previous_price = prev_record.prices[instrument_id]
            current_price = market_prices[instrument_id]
            if previous_price <= 0:
                continue
            predicted.append(prev_record.scores[instrument_id])
            realized.append(float((current_price - previous_price) / previous_price))

        if len(predicted) < 2:
            return

        ic = shadow_spearman_ic(predicted, realized)
        if not math.isnan(ic):
            self.ic_history.append(ic)
            rolling = self.rolling_ic
            log.debug(
                "shadow_scorer.ic_updated",
                daily_ic=round(ic, 4),
                rolling_ic=round(rolling, 4) if not math.isnan(rolling) else None,
            )


__all__ = ["ShadowDayRecord", "ShadowICTracker", "shadow_spearman_ic"]
