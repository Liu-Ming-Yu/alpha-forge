"""Stateful market-regime detector implementation."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.regime.classification import classify_regime, detector_version
from quant_platform.core.regime.states import (
    build_classified_state,
    build_no_stats_state,
    build_stable_state,
)
from quant_platform.core.regime.stats import MarketStats, compute_market_stats
from quant_platform.core.regime.thresholds import (
    DEFAULT_REGIME_THRESHOLDS,
    RegimeThresholds,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

log = structlog.get_logger(__name__)


class MarketRegimeDetector:
    """Stateful rule-based market regime classifier."""

    _BASE_VERSION = "v1.0-rule-based"
    is_simple_regime_detector = False

    def __init__(
        self,
        thresholds: RegimeThresholds | None = None,
        *,
        disagree_haircut: float = 0.75,
        log_updates: bool = True,
    ) -> None:
        self._thresholds = thresholds or DEFAULT_REGIME_THRESHOLDS
        if not 0.0 <= disagree_haircut <= 1.0:
            raise ValueError("disagree_haircut must be in [0, 1]")
        self._disagree_haircut = float(disagree_haircut)
        # ``update()`` emits a per-step ``regime_detector.updated`` debug
        # line. In live/paper trading that's once per cycle — useful
        # signal. In the research feature factory the detector is stepped
        # once per date across the whole universe-300 history (~hundreds
        # of thousands of calls), so the offline caller passes
        # ``log_updates=False`` to suppress it while live keeps the
        # default. State-machine behaviour is unaffected either way.
        self._log_updates = bool(log_updates)
        self._current_stats: MarketStats | None = None
        self._version = self._version_for(self._thresholds)
        self._stable_regime: RegimeLabel = RegimeLabel.TRANSITION
        self._regime_history: deque[RegimeLabel] = deque(maxlen=self._thresholds.stability_window)

    @classmethod
    def _version_for(cls, thresholds: RegimeThresholds) -> str:
        return detector_version(cls._BASE_VERSION, thresholds)

    @property
    def detector_version(self) -> str:
        return self._version

    def update(self, stats: MarketStats) -> None:
        """Update internal state with fresh market statistics."""
        self._current_stats = stats
        if self._log_updates:
            log.debug(
                "regime_detector.updated",
                trend_z=stats.trend_z,
                realized_vol=stats.realized_vol,
                breadth=stats.breadth,
            )

    def classify(self, stats: MarketStats) -> RegimeState:
        """Classify regime from explicit stats without changing detector state."""
        label, confidence = classify_regime(stats, self._thresholds)
        log.info(
            "regime_detector.classified",
            label=label.value,
            confidence=confidence,
            trend_z=stats.trend_z,
            realized_vol=stats.realized_vol,
            breadth=stats.breadth,
        )
        return build_classified_state(
            stats=stats,
            label=label,
            confidence=confidence,
            version=self._version,
            thresholds=self._thresholds,
        )

    def _classify_with_state(self, as_of: datetime) -> RegimeState:
        """Synchronous core of :meth:`detect` — advances the stability
        state machine and returns the stable :class:`RegimeState`.

        Extracted so both the async ``detect`` (live cycle steps) and
        the sync :meth:`step` (research backtest panels) go through the
        same deque-of-candidates / stable-regime / disagree-haircut
        logic — guaranteeing bit-for-bit label parity between live
        execution and research feature evidence.
        """
        if self._current_stats is None:
            log.warning(
                "regime_detector.no_stats",
                detail="no MarketStats loaded; returning TRANSITION with confidence=0",
            )
            return build_no_stats_state(as_of=as_of, version=self._version)

        stats_with_as_of = MarketStats(
            trend_z=self._current_stats.trend_z,
            realized_vol=self._current_stats.realized_vol,
            breadth=self._current_stats.breadth,
            as_of=as_of,
        )
        candidate_label, confidence = classify_regime(
            stats_with_as_of, self._thresholds, self._stable_regime
        )

        self._advance_stable_regime(candidate_label)
        return build_stable_state(
            stats=stats_with_as_of,
            label=self._stable_regime,
            candidate_label=candidate_label,
            confidence=confidence,
            version=self._version,
            stability_window=self._thresholds.stability_window,
            disagree_haircut=self._disagree_haircut,
        )

    async def detect(self, as_of: datetime) -> RegimeState:
        """Detect regime using the most recent ``update()`` stats.

        Async-compatible wrapper for the cycle-step interface; the
        actual work is synchronous and lives in
        :meth:`_classify_with_state`.
        """
        return self._classify_with_state(as_of)

    def step(self, stats: MarketStats, as_of: datetime) -> RegimeState:
        """Synchronous ``update(stats)`` + ``detect(as_of)`` in one call.

        Provided for callers that need detector parity without an async
        runtime — primarily the research feature factory, which walks
        an offline date series in order. Identical state-machine
        semantics to ``update()`` followed by ``await detect()`` (both
        go through :meth:`_classify_with_state`); see ADR-005 for why
        research must use exactly this entrypoint to claim live parity.
        """
        self.update(stats)
        return self._classify_with_state(as_of)

    def _advance_stable_regime(self, candidate_label: RegimeLabel) -> None:
        self._regime_history.append(candidate_label)
        window = self._thresholds.stability_window
        if len(self._regime_history) < window or len(set(self._regime_history)) != 1:
            return
        if self._stable_regime != candidate_label:
            log.info(
                "regime_detector.regime_change",
                previous=self._stable_regime.value,
                new=candidate_label.value,
                stability_window=window,
            )
        self._stable_regime = candidate_label

    @staticmethod
    def compute_stats(
        index_closes: list[float],
        instrument_closes: dict[uuid.UUID, list[float]],
        as_of: datetime,
        trend_window: int = 200,
        vol_window: int = 21,
        breadth_window: int = 50,
    ) -> MarketStats:
        """Compute ``MarketStats`` from daily close-price series."""
        return compute_market_stats(
            index_closes=index_closes,
            instrument_closes=instrument_closes,
            as_of=as_of,
            trend_window=trend_window,
            vol_window=vol_window,
            breadth_window=breadth_window,
        )


__all__ = ["MarketRegimeDetector"]
