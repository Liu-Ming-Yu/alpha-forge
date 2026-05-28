"""Artifact manifest checks for alpha promotion readiness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from quant_platform.application.features.admission import ordered_feature_schema_hash

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def text_manifest_checks(
    settings: PlatformSettings,
    *,
    manifest_payload: dict[str, Any],
    active_feature_set_version: str,
) -> list[dict[str, object]]:
    feature_names = tuple(str(name) for name in manifest_payload.get("feature_names", ()))
    manifest_weights = {
        str(name): float(value) for name, value in dict(manifest_payload.get("weights", {})).items()
    }
    active_weights = {
        str(name): float(value) for name, value in settings.llm.text_feature_weights.items()
    }
    expected_schema_hash = ordered_feature_schema_hash(feature_names)
    actual_schema_hash = str(manifest_payload.get("feature_schema_hash", ""))
    return [
        {
            "name": "text_manifest_signal_type",
            "passed": manifest_payload.get("signal_type") == "text",
            "detail": str(manifest_payload.get("signal_type", "")),
        },
        {
            "name": "text_manifest_feature_schema_hash",
            "passed": bool(feature_names) and actual_schema_hash == expected_schema_hash,
            "detail": actual_schema_hash or "missing feature_schema_hash",
        },
        {
            "name": "text_active_feature_set_version",
            "passed": active_feature_set_version
            == str(manifest_payload.get("feature_set_version", "")),
            "detail": active_feature_set_version or "no active model",
        },
        {
            "name": "text_prompt_version_matches",
            "passed": settings.llm.text_prompt_version
            == str(manifest_payload.get("prompt_version", "")),
            "detail": str(manifest_payload.get("prompt_version", "")),
        },
        {
            "name": "text_model_matches",
            "passed": settings.llm.model == str(manifest_payload.get("llm_model", "")),
            "detail": str(manifest_payload.get("llm_model", "")),
        },
        {
            "name": "text_weights_match_manifest",
            "passed": _weights_match(active_weights, manifest_weights),
            "detail": json.dumps(
                {"active": active_weights, "manifest": manifest_weights},
                sort_keys=True,
            ),
        },
    ]


def _weights_match(left: dict[str, float], right: dict[str, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(abs(float(left[name]) - float(right[name])) <= 1e-12 for name in left)
