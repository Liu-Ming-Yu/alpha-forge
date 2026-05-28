"""Feature-card and admission checks for live-LLM startup governance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import assert_feature_artifacts_admitted
from quant_platform.core.domain.production import PreflightCheck, ProductionProfile
from quant_platform.services.governance_service.llm_live_startup.helpers import (
    _feature_minimum_state,
    _object_mapping,
    _sha256_file,
    _string_sequence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings


def text_feature_card_checks(
    settings: PlatformSettings,
    *,
    manifest: Mapping[str, object],
) -> list[PreflightCheck]:
    """Return feature-card hash pinning checks for a text-model manifest."""
    feature_names = _string_sequence(manifest.get("feature_names"))
    manifest_hashes = {
        str(name): str(value)
        for name, value in _object_mapping(manifest.get("feature_card_hashes")).items()
    }
    manifest_paths = {
        str(name): str(value)
        for name, value in _object_mapping(manifest.get("feature_card_paths")).items()
    }
    missing: list[str] = []
    mismatched: list[str] = []
    for name in feature_names:
        expected = manifest_hashes.get(name, "")
        path = _feature_card_path(settings, name, manifest_paths)
        if not expected or not path.is_file():
            missing.append(name)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatched.append(name)
    return [
        PreflightCheck(
            name="llm_live_text_feature_cards_hash_pinned",
            passed=not missing and not mismatched and bool(feature_names),
            detail=json.dumps(
                {"missing": missing, "mismatched": mismatched, "features": feature_names},
                sort_keys=True,
            ),
        )
    ]


def text_feature_card_dir_deployment_checks(
    settings: PlatformSettings,
    *,
    profile: ProductionProfile,
) -> list[PreflightCheck]:
    """Return deployment-path checks for LLM live rehearsal."""
    if profile != ProductionProfile.LLM_LIVE_REHEARSAL:
        return []
    raw_dir = settings.llm.text_feature_card_dir.strip()
    path = Path(raw_dir).expanduser() if raw_dir else None
    return [
        PreflightCheck(
            name="llm_live_text_feature_card_dir_absolute",
            passed=bool(path is not None and path.is_absolute()),
            detail=(
                "QP__LLM__TEXT_FEATURE_CARD_DIR must be an absolute deployment path "
                f"for llm_live_rehearsal; configured={raw_dir!r}"
            ),
        )
    ]


def text_feature_admission_check(
    settings: PlatformSettings,
    *,
    manifest: Mapping[str, object],
    profile: ProductionProfile,
) -> PreflightCheck:
    """Return the feature-audit admission check for manifest features."""
    feature_names = _string_sequence(manifest.get("feature_names"))
    feature_versions = dict(settings.llm.text_feature_versions)
    if not feature_versions:
        feature_versions = {name: settings.llm.text_feature_set_version for name in feature_names}
    minimum_state = _feature_minimum_state(profile)
    try:
        assert_feature_artifacts_admitted(
            audit_root=Path(settings.storage.object_store_root),
            feature_names=feature_names,
            feature_versions=feature_versions,
            feature_set_version=settings.llm.text_feature_set_version,
            minimum_state=minimum_state,
            model_feature_schema_hash=str(manifest.get("feature_schema_hash", "")),
        )
    except Exception as exc:
        return PreflightCheck(
            name="llm_live_text_feature_audits_admitted",
            passed=False,
            detail=str(exc),
        )
    return PreflightCheck(
        name="llm_live_text_feature_audits_admitted",
        passed=True,
        detail=f"features={list(feature_names)} minimum_state={minimum_state.value}",
    )


def _feature_card_path(
    settings: PlatformSettings,
    feature_name: str,
    manifest_paths: Mapping[str, str],
) -> Path:
    raw_manifest_path = manifest_paths.get(feature_name, "").strip()
    if raw_manifest_path:
        return Path(raw_manifest_path).expanduser()
    raw_dir = settings.llm.text_feature_card_dir.strip()
    if raw_dir:
        return Path(raw_dir).expanduser() / f"{feature_name}.json"
    return (
        Path("infra")
        / "config"
        / "feature_cards"
        / settings.llm.text_feature_set_version
        / f"{feature_name}.json"
    )
