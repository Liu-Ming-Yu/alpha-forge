"""Signal-combination utilities for portfolio construction."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.signals import SignalScore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

log = structlog.get_logger(__name__)


class ICWeightedSignalCombiner:
    """Combine per-factor signal scores with exponentially weighted IC weights."""

    def __init__(
        self,
        ic_history: Mapping[str, Sequence[tuple[datetime, float]]],
        *,
        ewma_span: int = 10,
        lookback: int = 60,
        min_weight: float = 0.01,
    ) -> None:
        self._ic_history = ic_history
        self._alpha = 2.0 / (ewma_span + 1)
        self._lookback = lookback
        self._min_weight = min_weight

    def factor_weights(self) -> dict[str, float]:
        """Compute normalized EWMA-IC weights for each factor."""
        raw: dict[str, float] = {}
        for factor, observations in self._ic_history.items():
            recent = list(observations)[-self._lookback :]
            if not recent:
                raw[factor] = 0.0
                continue
            ewma = 0.0
            for _, ic in recent:
                ewma = self._alpha * ic + (1 - self._alpha) * ewma
            raw[factor] = max(0.0, ewma)

        total = sum(raw.values())
        if total < 1e-9:
            n = len(raw) or 1
            return {f: 1.0 / n for f in raw}

        weights = {f: w / total for f, w in raw.items()}
        filtered = {f: w for f, w in weights.items() if w >= self._min_weight}
        total_f = sum(filtered.values())
        if total_f < 1e-9:
            return weights
        return {f: w / total_f for f, w in filtered.items()}

    def combine(
        self,
        factor_signals: Mapping[str, list[SignalScore]],
        as_of: datetime,
        strategy_run_id: uuid.UUID,
    ) -> list[SignalScore]:
        """Blend per-factor signal scores into one combined score per instrument."""
        weights = self.factor_weights()
        instrument_scores: dict[uuid.UUID, float] = {}
        instrument_confidences: dict[uuid.UUID, list[float]] = {}
        seen_vector: dict[uuid.UUID, uuid.UUID] = {}

        for factor, signals in factor_signals.items():
            w = weights.get(factor, 0.0)
            if w < 1e-9:
                continue
            for sig in signals:
                iid = sig.instrument_id
                instrument_scores[iid] = instrument_scores.get(iid, 0.0) + w * sig.score
                instrument_confidences.setdefault(iid, []).append(sig.confidence)
                seen_vector.setdefault(iid, sig.feature_vector_id)

        model_ver = "ic_weighted_combiner_v1"
        blended: list[SignalScore] = []
        for iid, blended_score in instrument_scores.items():
            clipped = max(-1.0, min(1.0, blended_score))
            confs = instrument_confidences.get(iid, [1.0])
            blended.append(
                SignalScore(
                    score_id=uuid.uuid4(),
                    instrument_id=iid,
                    strategy_run_id=strategy_run_id,
                    as_of=as_of,
                    score=clipped,
                    confidence=sum(confs) / len(confs),
                    model_version=model_ver,
                    feature_vector_id=seen_vector[iid],
                )
            )

        log.info(
            "portfolio.ic_weighted_combine",
            n_instruments=len(blended),
            factor_weights={f: round(w, 4) for f, w in weights.items()},
        )
        return blended
