"""Text-model manifest checks for live-LLM startup governance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.application.research.text_model_manifest import (
    TEXT_MODEL_MANIFEST_SCHEMA_VERSION,
    read_text_model_manifest,
)
from quant_platform.core.domain.production import PreflightCheck
from quant_platform.services.governance_service.llm_live_startup.helpers import (
    _coerce_float,
    _object_mapping,
    _parse_timestamp,
    _string_sequence,
)
from quant_platform.services.governance_service.llm_live_startup.paths import (
    expected_text_feature_schema_hash,
    text_model_manifest_path,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.config import PlatformSettings


@dataclass(frozen=True)
class TextModelManifestEvidence:
    """Loaded text-model manifest and the path that anchored the check."""

    path: Path
    payload: Mapping[str, object]


def load_text_model_manifest(
    settings: PlatformSettings,
) -> tuple[TextModelManifestEvidence | None, PreflightCheck]:
    """Load the configured text-model manifest and return its presence check."""
    manifest_path = text_model_manifest_path(settings)
    if manifest_path is None:
        return (
            None,
            PreflightCheck(
                name="llm_live_text_model_manifest_present",
                passed=False,
                detail="QP__LLM__TEXT_MODEL_MANIFEST is required",
            ),
        )
    if not manifest_path.is_file():
        return (
            None,
            PreflightCheck(
                name="llm_live_text_model_manifest_present",
                passed=False,
                detail=f"text model manifest not found: {manifest_path}",
            ),
        )

    try:
        manifest = read_text_model_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            None,
            PreflightCheck(
                name="llm_live_text_model_manifest_present",
                passed=False,
                detail=f"invalid text model manifest {manifest_path}: {exc}",
            ),
        )

    return (
        TextModelManifestEvidence(path=manifest_path, payload=manifest),
        PreflightCheck(
            name="llm_live_text_model_manifest_present",
            passed=True,
            detail=str(manifest_path),
        ),
    )


def text_manifest_policy_checks(
    settings: PlatformSettings,
    *,
    manifest: Mapping[str, object],
    as_of: datetime,
) -> list[PreflightCheck]:
    """Return policy checks for a loaded text-model manifest."""
    feature_names = _string_sequence(manifest.get("feature_names"))
    expected_schema_hash = ordered_feature_schema_hash(feature_names)
    active_names = tuple(settings.llm.text_feature_weights)
    active_weights = {
        str(name): float(value) for name, value in settings.llm.text_feature_weights.items()
    }
    manifest_weights = {
        str(name): _coerce_float(value)
        for name, value in _object_mapping(manifest.get("weights")).items()
    }
    created_at = _parse_timestamp(str(manifest.get("created_at", "")))
    fresh = created_at >= as_of.astimezone(UTC) - timedelta(
        days=settings.llm.live_evidence_stale_after_days
    )
    return [
        PreflightCheck(
            name="llm_live_text_manifest_schema_v2",
            passed=manifest.get("schema_version") == TEXT_MODEL_MANIFEST_SCHEMA_VERSION,
            detail=str(manifest.get("schema_version", "")),
        ),
        PreflightCheck(
            name="llm_live_text_manifest_fresh",
            passed=fresh,
            detail=(
                f"created_at={created_at.isoformat()} "
                f"stale_after_days={settings.llm.live_evidence_stale_after_days}"
            ),
        ),
        PreflightCheck(
            name="llm_live_text_manifest_matches_settings",
            passed=(
                manifest.get("signal_type") == "text"
                and manifest.get("provider") == settings.llm.provider
                and manifest.get("llm_model") == settings.llm.model
                and manifest.get("prompt_version") == settings.llm.text_prompt_version
                and manifest.get("feature_set_version") == settings.llm.text_feature_set_version
                and feature_names == active_names
                and manifest_weights == active_weights
            ),
            detail=json.dumps(
                {
                    "manifest_provider": manifest.get("provider"),
                    "manifest_model": manifest.get("llm_model"),
                    "manifest_prompt": manifest.get("prompt_version"),
                    "manifest_feature_set": manifest.get("feature_set_version"),
                    "feature_names": feature_names,
                },
                sort_keys=True,
                default=str,
            ),
        ),
        PreflightCheck(
            name="llm_live_text_manifest_schema_hash",
            passed=(
                bool(feature_names)
                and manifest.get("feature_schema_hash") == expected_schema_hash
                and expected_schema_hash == expected_text_feature_schema_hash(settings)
            ),
            detail=str(manifest.get("feature_schema_hash", "")),
        ),
    ]
