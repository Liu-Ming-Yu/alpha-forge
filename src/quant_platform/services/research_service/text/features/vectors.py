"""FeatureVector construction helpers for text extraction."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.research import FeatureVector
from quant_platform.services.research_service.text.extraction.sec_primary_compaction import (
    compact_sec_primary_text,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.market_data.text_events import TextEvent

log = structlog.get_logger(__name__)


def prepare_text_content(
    event: TextEvent,
    text_content: str,
    *,
    prompt_version: str,
) -> tuple[str, str, dict[str, object] | None]:
    raw_digest = hashlib.sha256(text_content.encode("utf-8")).hexdigest()[:32]
    if prompt_version != "v5":
        return text_content, raw_digest, None
    compacted = compact_sec_primary_text(
        text_content,
        form_type=str((event.metadata or {}).get("form_type", "")),
    )
    compacted_digest = hashlib.sha256(compacted.text.encode("utf-8")).hexdigest()[:32]
    lineage = compacted.to_payload(
        raw_source_uri=event.source_uri,
        raw_artifact_uri=event.artifact_uri,
        raw_content_digest=raw_digest,
        compacted_content_digest=compacted_digest,
    )
    _safe_log(
        "info",
        "text_extractor.compacted",
        event_id=str(event.event_id),
        prompt_version=prompt_version,
        policy=compacted.policy_name,
        original_chars=compacted.original_chars,
        compacted_chars=compacted.compacted_chars,
        selected_section_labels=list(compacted.selected_section_labels),
    )
    return compacted.text, compacted_digest, lineage


def build_text_feature_vector(
    *,
    event: TextEvent,
    strategy_run_id: uuid.UUID,
    as_of: datetime | None,
    features: Mapping[str, float],
    feature_set_version: str,
    artifact_uri: str,
) -> FeatureVector:
    vector_as_of = as_of if as_of is not None else event.occurred_at
    return FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=event.instrument_id or uuid.UUID(int=0),
        strategy_run_id=strategy_run_id,
        as_of=vector_as_of,
        features=dict(features),
        feature_set_version=feature_set_version,
        artifact_uri=artifact_uri,
        available_at=vector_as_of,
    )


def _safe_log(level: str, event: str, **kwargs: object) -> None:
    try:
        getattr(log, level)(event, **kwargs)
    except OSError:
        return
