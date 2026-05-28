"""Per-event shadow text scoring helpers."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.core.domain.production import SignalContribution
from quant_platform.core.domain.signals import SignalScore
from quant_platform.services.research_service.text.features import (
    FeatureExtractionError,
    LLMTextFeatureExtractor,
)
from quant_platform.services.research_service.text.features.prompts import MAX_TEXT_CHARS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.core.domain.research import FeatureVector, StrategyRun

log = structlog.get_logger(__name__)


class _TextExtractor(Protocol):
    def extract(
        self,
        event: TextEvent,
        text_content: str,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime | None = None,
    ) -> FeatureVector: ...


@dataclass(frozen=True)
class ShadowTextEventResult:
    instrument_id: uuid.UUID
    score: float
    signal: SignalScore
    contribution: SignalContribution


def blend_text_score(features: Mapping[str, float]) -> float:
    """Combine text sentiment and guidance into the shadow score."""
    raw_sentiment = float(features.get("text_sentiment", 0.0))
    guidance = float(features.get("guidance_direction", 0.0))
    combined = 0.70 * raw_sentiment + 0.30 * guidance
    return max(-1.0, min(1.0, combined))


async def process_shadow_text_event(
    *,
    extractor: LLMTextFeatureExtractor | _TextExtractor,
    feature_repo: FeatureRepository,
    factor_version: str,
    event: TextEvent,
    content: str,
    strategy_run: StrategyRun,
    as_of: datetime,
) -> ShadowTextEventResult | None:
    """Extract, persist, and score one shadow text event."""
    if event.instrument_id is None:
        return None
    if not content:
        log.debug("shadow_scorer.no_content", event_id=str(event.event_id))
        return None

    source_hash = hashlib.sha256(content.encode()).hexdigest()
    truncated = len(content) > MAX_TEXT_CHARS
    try:
        vector = extractor.extract(event, content, strategy_run.run_id, as_of=as_of)
    except FeatureExtractionError as exc:
        log.warning(
            "shadow_scorer.extraction_failed",
            event_id=str(event.event_id),
            instrument_id=str(event.instrument_id),
            error=str(exc),
        )
        return None

    vector = replace(
        vector,
        metadata={
            "llm_model": getattr(extractor, "_model", "unknown"),
            "prompt_version": getattr(extractor, "_prompt_version", "unknown"),
            "truncated": str(truncated),
            "source_text_hash": source_hash,
        },
    )

    try:
        await feature_repo.store_vector(vector)
    except Exception as exc:
        log.warning(
            "shadow_scorer.store_failed",
            vector_id=str(vector.vector_id),
            error=str(exc),
        )

    combined = blend_text_score(vector.features)
    signal = SignalScore(
        score_id=uuid.uuid4(),
        instrument_id=event.instrument_id,
        strategy_run_id=strategy_run.run_id,
        as_of=as_of,
        score=combined,
        confidence=float(vector.features.get("revenue_revision_magnitude", 0.5)),
        model_version=factor_version,
        feature_vector_id=vector.vector_id,
    )
    contribution = SignalContribution(
        contribution_id=uuid.uuid4(),
        score_id=signal.score_id,
        strategy_run_id=strategy_run.run_id,
        instrument_id=event.instrument_id,
        as_of=as_of,
        source="text",
        source_model_version=signal.model_version,
        raw_score=combined,
        normalized_score=combined,
        blend_weight=0.0,
        confidence=signal.confidence,
        feature_vector_id=vector.vector_id,
        promotion_state="shadow",
    )
    return ShadowTextEventResult(
        instrument_id=event.instrument_id,
        score=combined,
        signal=signal,
        contribution=contribution,
    )
