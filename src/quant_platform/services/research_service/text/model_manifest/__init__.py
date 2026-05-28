"""Compatibility exports for governed text alpha model manifests."""

from __future__ import annotations

from quant_platform.application.research.text_model_manifest import (
    TEXT_MODEL_MANIFEST_SCHEMA_VERSION,
    TextModelManifest,
    read_text_model_manifest,
    write_text_model_manifest,
)

__all__ = [
    "TEXT_MODEL_MANIFEST_SCHEMA_VERSION",
    "TextModelManifest",
    "read_text_model_manifest",
    "write_text_model_manifest",
]
