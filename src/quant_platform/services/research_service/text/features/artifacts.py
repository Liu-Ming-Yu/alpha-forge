"""Artifact helpers for governed text feature extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.text.features.errors import FeatureExtractionError
from quant_platform.services.research_service.text.features.prompts import PROMPTS
from quant_platform.services.research_service.text.features.validation import (
    validate_text_features,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.core.domain.market_data.text_events import TextEvent


def extraction_artifact_path(
    *,
    artifact_root: Path | None,
    provider: str,
    model: str,
    prompt_version: str,
    event: TextEvent,
    content_digest: str,
) -> Path | None:
    if artifact_root is None:
        return None
    event_dir = artifact_root / provider / model / prompt_version
    return event_dir / f"{event.event_id}_{content_digest}.json"


def write_extraction_artifact(
    *,
    artifact_root: Path | None,
    provider: str,
    model: str,
    prompt_version: str,
    event: TextEvent,
    content_digest: str,
    source_artifact_uri: str,
    features: dict[str, float],
    raw_response: str,
    lineage: Mapping[str, object] | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
) -> str:
    path = extraction_artifact_path(
        artifact_root=artifact_root,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        event=event,
        content_digest=content_digest,
    )
    if path is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "event_id": str(event.event_id),
        "event_type": event.event_type.value,
        "instrument_id": str(event.instrument_id) if event.instrument_id else None,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "system_prompt": PROMPTS[prompt_version],
        "source_artifact_uri": source_artifact_uri,
        "content_digest": content_digest,
        "raw_response": raw_response,
        "features": features,
    }
    if lineage is not None:
        payload["lineage"] = dict(lineage)
    if runtime_metadata is not None:
        payload["runtime_metadata"] = dict(runtime_metadata)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_extraction_artifact_features(
    *,
    artifact_root: Path | None,
    provider: str,
    model: str,
    prompt_version: str,
    event: TextEvent,
    content_digest: str,
) -> tuple[dict[str, float], Path] | None:
    path = extraction_artifact_path(
        artifact_root=artifact_root,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        event=event,
        content_digest=content_digest,
    )
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_features = payload.get("features")
    if not isinstance(raw_features, dict):
        raise FeatureExtractionError("text extraction artifact missing features object")
    return validate_text_features(raw_features, prompt_version=prompt_version), path


__all__ = [
    "extraction_artifact_path",
    "load_extraction_artifact_features",
    "write_extraction_artifact",
]
