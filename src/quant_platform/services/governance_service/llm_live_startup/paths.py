"""Paths and schema helpers for live-LLM startup governance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def llm_extraction_artifact_root(settings: PlatformSettings) -> Path:
    raw = settings.llm.extraction_artifact_root.strip()
    if raw:
        return Path(raw).expanduser()
    return Path(settings.storage.object_store_root) / "research" / "text_events" / "extractions"


def llm_live_startup_assertion_path(settings: PlatformSettings) -> Path:
    raw = settings.llm.live_startup_assertion_path.strip()
    if raw:
        return Path(raw).expanduser()
    return (
        Path(settings.storage.object_store_root) / "governance" / "llm_live_startup_assertion.json"
    )


def text_model_manifest_path(settings: PlatformSettings) -> Path | None:
    raw = settings.llm.text_model_manifest.strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def expected_text_feature_schema_hash(settings: PlatformSettings) -> str:
    return ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights))
