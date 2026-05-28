"""Shadow text scorer for the LLM text feature layer."""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.production import PredictionResult
from quant_platform.services.research_service.text.features.validation import FEATURE_KEYS
from quant_platform.services.research_service.text.shadow.cycle import (
    process_shadow_text_event,
)
from quant_platform.services.research_service.text.shadow.ic_tracker import (
    ShadowDayRecord as _DayRecord,
)
from quant_platform.services.research_service.text.shadow.ic_tracker import (
    ShadowICTracker,
)
from quant_platform.services.research_service.text.shadow.ic_tracker import (
    shadow_spearman_ic as _spearman_ic,
)

if TYPE_CHECKING:
    from decimal import Decimal

    from quant_platform.core.contracts import (
        FeatureRepository,
        PredictionEvidenceRepository,
        SignalContributionRepository,
    )
    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.core.domain.production import SignalContribution
    from quant_platform.core.domain.research import StrategyRun
    from quant_platform.core.domain.signals import SignalScore
    from quant_platform.services.research_service.text.features import LLMTextFeatureExtractor

log = structlog.get_logger(__name__)

TEXT_SHADOW_FEATURE_SCHEMA_HASH = hashlib.sha256(
    f"text-shadow-v1:{','.join(FEATURE_KEYS)}".encode()
).hexdigest()


class ShadowTextScorer:
    """Runs text feature extraction and shadow scoring each strategy cycle."""

    FACTOR_VERSION = "text-shadow-v1"

    def __init__(
        self,
        extractor: LLMTextFeatureExtractor,
        feature_repo: FeatureRepository,
        *,
        ic_window: int = 20,
        min_ic: float = 0.05,
        min_observations: int = 20,
        contribution_repo: SignalContributionRepository | None = None,
        prediction_evidence_repo: PredictionEvidenceRepository | None = None,
        horizon: str = "21d",
    ) -> None:
        self._extractor = extractor
        self._feature_repo = feature_repo
        self._contribution_repo = contribution_repo
        self._prediction_evidence_repo = prediction_evidence_repo
        self._horizon = horizon
        self._ic_window = ic_window
        self._min_ic = min_ic
        self._min_observations = min_observations
        self._ic_tracker = ShadowICTracker(ic_window=ic_window)
        # Compatibility: integration tests and historical notebooks inspect these.
        self._score_history = self._ic_tracker.score_history
        self._ic_history = self._ic_tracker.ic_history

    async def score_cycle(
        self,
        events: list[TextEvent],
        text_contents: dict[uuid.UUID, str],
        strategy_run: StrategyRun,
        as_of: datetime,
        market_prices: dict[uuid.UUID, Decimal] | None = None,
    ) -> list[SignalScore]:
        """Extract text features and compute shadow scores."""
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        if market_prices and len(self._score_history) > 0:
            self._ic_tracker.update_ic(market_prices)

        cycle_scores: dict[uuid.UUID, float] = {}
        shadow_signals: list[SignalScore] = []
        contributions: list[SignalContribution] = []
        predictions: list[PredictionResult] = []

        for event in events:
            if event.instrument_id is None:
                continue
            content = text_contents.get(event.event_id, "")
            result = await process_shadow_text_event(
                extractor=self._extractor,
                feature_repo=self._feature_repo,
                factor_version=self.FACTOR_VERSION,
                event=event,
                content=content,
                strategy_run=strategy_run,
                as_of=as_of,
            )
            if result is None:
                continue
            cycle_scores[result.instrument_id] = result.score
            shadow_signals.append(result.signal)
            contributions.append(result.contribution)
            predictions.append(
                PredictionResult(
                    prediction_id=uuid.uuid4(),
                    strategy_run_id=strategy_run.run_id,
                    instrument_id=result.instrument_id,
                    source="text",
                    model_version=result.signal.model_version,
                    as_of=result.signal.as_of,
                    horizon=self._horizon,
                    expected_return=result.score,
                    rank_score=result.score,
                    confidence=result.signal.confidence,
                    feature_schema_hash=TEXT_SHADOW_FEATURE_SCHEMA_HASH,
                    calibration_bucket="shadow:daily:text",
                    metadata={"feature_set_version": self.FACTOR_VERSION},
                )
            )

        if contributions and self._contribution_repo is not None:
            await self._save_contributions(contributions)
        if predictions and self._prediction_evidence_repo is not None:
            await self._save_predictions(predictions)

        self._ic_tracker.record_scores(
            as_of=as_of,
            scores=cycle_scores,
            market_prices=market_prices,
        )

        rolling_ic = self.rolling_ic
        log.info(
            "shadow_scorer.cycle_complete",
            shadow_scores=len(shadow_signals),
            rolling_ic_20d=round(rolling_ic, 4) if not math.isnan(rolling_ic) else None,
            ic_obs=len(self._ic_history),
            as_of=str(as_of),
        )

        self._maybe_emit_ic_gauge(rolling_ic)
        return shadow_signals

    @property
    def rolling_ic(self) -> float:
        """Rolling mean IC over the last ``ic_window`` days."""
        return self._ic_tracker.rolling_ic

    @property
    def ic_observations(self) -> int:
        """Number of daily IC observations accumulated so far."""
        return self._ic_tracker.observations

    def passes_ic_gate(
        self,
        min_ic: float | None = None,
        min_observations: int | None = None,
    ) -> bool:
        """Return True if rolling IC satisfies the promotion gate."""
        effective_min_ic = min_ic if min_ic is not None else self._min_ic
        effective_min_obs = (
            min_observations if min_observations is not None else self._min_observations
        )
        if self.ic_observations < effective_min_obs:
            return False
        ic = self.rolling_ic
        if math.isnan(ic):
            return False
        return ic >= effective_min_ic

    async def _save_contributions(self, contributions: list[SignalContribution]) -> None:
        if self._contribution_repo is None:
            return
        try:
            await self._contribution_repo.save_signal_contributions(contributions)
        except Exception as exc:
            log.warning("shadow_scorer.contribution_store_failed", error=str(exc))

    async def _save_predictions(self, predictions: list[PredictionResult]) -> None:
        if self._prediction_evidence_repo is None:
            return
        try:
            for prediction in predictions:
                await self._prediction_evidence_repo.save_prediction_result(prediction)
        except Exception as exc:
            log.warning("shadow_scorer.prediction_store_failed", error=str(exc))

    def _maybe_emit_ic_gauge(self, ic: float) -> None:
        """Set Prometheus gauge if the metrics module is available."""
        try:
            from quant_platform.telemetry.metrics import (
                TEXT_SIGNAL_IC_ROLLING_20D,
            )

            if not math.isnan(ic):
                TEXT_SIGNAL_IC_ROLLING_20D.set(ic)
        except (ImportError, AttributeError):
            pass


__all__ = [
    "ShadowTextScorer",
    "TEXT_SHADOW_FEATURE_SCHEMA_HASH",
    "_DayRecord",
    "_spearman_ic",
]
