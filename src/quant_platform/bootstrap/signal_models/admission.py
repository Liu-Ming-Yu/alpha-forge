"""Promoted alpha-source admission checks for runtime signal models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import (
    assert_feature_artifacts_admitted,
    ordered_feature_schema_hash,
)
from quant_platform.core.domain.research import FeatureProductionState
from quant_platform.infrastructure.postgres.row_coercion import require_mapping, require_sequence

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.config import PlatformSettings


def assert_promoted_alpha_sources_configured(settings: PlatformSettings) -> None:
    if not settings.alpha.require_promotion_gate:
        return
    weights = settings.alpha.source_weights
    mode = settings.alpha.ensemble_mode
    if mode not in {"paper", "live"}:
        return
    if weights.get("xgboost", 0.0) > 0 and not settings.boosting.artifact_manifest:
        raise RuntimeError(
            f"QP__ALPHA__ENSEMBLE_MODE={mode} requires "
            "QP__BOOSTING__ARTIFACT_MANIFEST when xgboost has positive weight. "
            "Run quant-platform alpha assert before enabling the promoted source."
        )
    if weights.get("text", 0.0) > 0 and not (
        settings.llm.shadow_mode_enabled or settings.llm.live_mode_enabled
    ):
        raise RuntimeError(
            f"QP__ALPHA__ENSEMBLE_MODE={mode} requires text shadow/live scoring "
            "to be enabled when text has positive weight. Run quant-platform "
            "alpha assert before enabling the promoted source."
        )
    if weights.get("text", 0.0) > 0:
        _assert_promoted_feature_source_admitted(
            settings,
            source="text",
            feature_names=tuple(settings.llm.text_feature_weights),
            feature_versions=dict(settings.llm.text_feature_versions),
            feature_set_version=_source_admission_feature_set_version(
                settings,
                settings.llm.text_feature_set_version,
            ),
            default_feature_version=settings.llm.text_feature_set_version,
        )
    if weights.get("event", 0.0) > 0:
        _assert_promoted_feature_source_admitted(
            settings,
            source="event",
            feature_names=tuple(settings.alpha.event_feature_weights),
            feature_versions=dict(settings.alpha.event_feature_versions),
            feature_set_version=_source_admission_feature_set_version(
                settings,
                settings.alpha.event_feature_set_version,
            ),
            default_feature_version=settings.alpha.event_feature_set_version,
        )
    if weights.get("intraday", 0.0) > 0:
        _assert_promoted_feature_source_admitted(
            settings,
            source="intraday",
            feature_names=tuple(settings.alpha.intraday_feature_weights),
            feature_versions=dict(settings.alpha.intraday_feature_versions),
            feature_set_version=_source_admission_feature_set_version(
                settings,
                settings.alpha.intraday_feature_set_version,
            ),
            default_feature_version=settings.alpha.intraday_feature_set_version,
        )


def assert_promoted_boosting_features_admitted(
    settings: PlatformSettings,
    manifest: Mapping[str, object],
) -> None:
    feature_names = tuple(
        str(name)
        for name in require_sequence(
            manifest["feature_names"],
            name="boosting_manifest.feature_names",
        )
    )
    feature_versions = {
        str(name): str(version)
        for name, version in require_mapping(
            manifest.get("feature_versions", {}),
            name="boosting_manifest.feature_versions",
        ).items()
    }
    missing_versions = [name for name in feature_names if name not in feature_versions]
    if missing_versions:
        missing = ", ".join(sorted(missing_versions))
        raise RuntimeError(
            "boosting manifest feature_versions is required for promoted "
            f"paper/live admission; missing: {missing}"
        )
    feature_versions = _resolved_boosting_feature_versions(
        settings,
        feature_names=feature_names,
        feature_versions=feature_versions,
        feature_set_version=str(manifest["feature_set_version"]),
    )
    assert_feature_artifacts_admitted(
        audit_root=Path(settings.storage.object_store_root),
        feature_names=feature_names,
        feature_versions=feature_versions,
        feature_set_version=str(manifest["feature_set_version"]),
        minimum_state=_minimum_feature_state(settings),
        model_feature_schema_hash=str(manifest["feature_schema_hash"]),
    )


def load_boosting_manifest_policy_payload(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read boosting artifact manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid boosting artifact manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"boosting artifact manifest must be a JSON object: {path}")
    feature_names = list(raw.get("feature_names", []))
    if not feature_names or any(not isinstance(name, str) for name in feature_names):
        raise RuntimeError("boosting manifest feature_names must be a non-empty string list")
    expected_hash = ordered_feature_schema_hash(feature_names)
    if raw.get("feature_schema_hash") != expected_hash:
        raise RuntimeError("boosting manifest feature_schema_hash does not match feature_names")
    feature_versions = raw.get("feature_versions", {})
    if feature_versions is not None and not isinstance(feature_versions, dict):
        raise RuntimeError("boosting manifest feature_versions must be an object")
    raw["feature_names"] = feature_names
    raw["feature_versions"] = dict(require_mapping(feature_versions or {}, name="feature_versions"))
    return raw


def promoted_weight(settings: PlatformSettings, source: str) -> float:
    return float(settings.alpha.source_weights.get(source, 0.0))


def _assert_promoted_feature_source_admitted(
    settings: PlatformSettings,
    *,
    source: str,
    feature_names: tuple[str, ...],
    feature_versions: Mapping[str, str],
    feature_set_version: str,
    default_feature_version: str,
) -> None:
    if not feature_names:
        raise RuntimeError(
            f"promoted {source} source requires configured feature weights to name "
            "at least one audited feature"
        )
    feature_versions = _feature_versions_with_default(
        feature_names,
        feature_versions,
        default_feature_version=default_feature_version,
    )
    assert_feature_artifacts_admitted(
        audit_root=Path(settings.storage.object_store_root),
        feature_names=feature_names,
        feature_versions=feature_versions,
        feature_set_version=feature_set_version,
        minimum_state=_minimum_feature_state(settings),
        model_feature_schema_hash=ordered_feature_schema_hash(feature_names),
    )


def _minimum_feature_state(settings: PlatformSettings) -> FeatureProductionState:
    if (
        settings.alpha.ensemble_mode == "live"
        and settings.llm.live_mode_enabled
        and settings.llm.live_rehearsal_enabled
        and settings.broker.paper_trading
    ):
        return FeatureProductionState.PAPER
    if settings.alpha.ensemble_mode == "live":
        return FeatureProductionState.LIVE
    return FeatureProductionState.PAPER


def _source_admission_feature_set_version(
    settings: PlatformSettings,
    source_feature_set_version: str,
) -> str:
    if settings.alpha.ensemble_mode != "paper":
        return source_feature_set_version
    promoted = str(settings.alpha.promoted_feature_set_version or "").strip()
    if not promoted:
        return source_feature_set_version
    if _uses_composite_paper_admission(settings):
        return promoted
    return source_feature_set_version


def _uses_composite_paper_admission(settings: PlatformSettings) -> bool:
    weights = settings.alpha.source_weights
    promoted_sources = {
        source
        for source in ("xgboost", "text", "event", "intraday")
        if float(weights.get(source, 0.0)) > 0.0
    }
    return len(promoted_sources) > 1 or "xgboost" in promoted_sources


def _feature_versions_with_default(
    feature_names: tuple[str, ...],
    feature_versions: Mapping[str, str],
    *,
    default_feature_version: str,
) -> dict[str, str]:
    return {
        name: str(feature_versions.get(name, default_feature_version)) for name in feature_names
    }


def _resolved_boosting_feature_versions(
    settings: PlatformSettings,
    *,
    feature_names: tuple[str, ...],
    feature_versions: Mapping[str, str],
    feature_set_version: str,
) -> dict[str, str]:
    source_versions = _configured_source_feature_versions(settings, feature_names)
    promoted = str(settings.alpha.promoted_feature_set_version or "").strip()
    resolved: dict[str, str] = {}
    for name in feature_names:
        version = str(feature_versions[name])
        if (
            settings.alpha.ensemble_mode == "paper"
            and promoted
            and feature_set_version == promoted
            and version == feature_set_version
            and name in source_versions
        ):
            version = source_versions[name]
        resolved[name] = version
    return resolved


def _configured_source_feature_versions(
    settings: PlatformSettings,
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    source_maps = (
        (
            settings.llm.text_feature_weights,
            settings.llm.text_feature_versions,
            settings.llm.text_feature_set_version,
        ),
        (
            settings.alpha.event_feature_weights,
            settings.alpha.event_feature_versions,
            settings.alpha.event_feature_set_version,
        ),
        (
            settings.alpha.intraday_feature_weights,
            settings.alpha.intraday_feature_versions,
            settings.alpha.intraday_feature_set_version,
        ),
    )
    wanted = set(feature_names)
    for weights, explicit_versions, default_version in source_maps:
        for name in weights:
            if name in wanted:
                resolved[str(name)] = str(explicit_versions.get(name, default_version))
    return resolved


__all__ = [
    "assert_promoted_alpha_sources_configured",
    "assert_promoted_boosting_features_admitted",
    "load_boosting_manifest_policy_payload",
    "promoted_weight",
]
