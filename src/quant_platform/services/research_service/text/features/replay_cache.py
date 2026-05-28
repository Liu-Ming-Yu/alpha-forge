"""Replay/cache helpers for LLM text feature extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.text.features.artifacts import (
    load_extraction_artifact_features,
)
from quant_platform.services.research_service.text.features.errors import FeatureExtractionError
from quant_platform.services.research_service.text.features.vectors import build_text_feature_vector

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.core.domain.research import FeatureVector

    CacheKey = tuple[uuid.UUID, str, str, str, str]
    SafeLog = Callable[..., None]


def load_cached_extraction_vector(
    *,
    artifact_root: Path | None,
    provider: str,
    model: str,
    prompt_version: str,
    event: TextEvent,
    content_digest: str,
    strategy_run_id: uuid.UUID,
    as_of: datetime | None,
    cache_key: CacheKey,
    cache: dict[CacheKey, FeatureVector],
    safe_log: SafeLog,
) -> FeatureVector | None:
    """Load a cached artifact and return a feature vector when available."""
    try:
        cached = load_extraction_artifact_features(
            artifact_root=artifact_root,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            event=event,
            content_digest=content_digest,
        )
        if cached is None:
            return None
        features, path = cached
    except (FeatureExtractionError, OSError, json.JSONDecodeError) as exc:
        safe_log(
            "warning",
            "text_extractor.cache_artifact_ignored",
            event_id=str(event.event_id),
            artifact=str(artifact_root or ""),
            error=str(exc),
        )
        return None
    artifact_uri = f"{path}#prompt={prompt_version}"
    vector = build_text_feature_vector(
        event=event,
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        features=features,
        feature_set_version=f"text-{prompt_version}",
        artifact_uri=artifact_uri,
    )
    cache[cache_key] = vector
    safe_log(
        "info",
        "text_extractor.cache_artifact_hit",
        event_id=str(event.event_id),
        event_type=event.event_type.value,
        instrument_id=str(event.instrument_id) if event.instrument_id else None,
        sentiment=features.get("text_sentiment"),
        guidance=features.get("guidance_direction"),
    )
    return vector


__all__ = ["load_cached_extraction_vector"]
