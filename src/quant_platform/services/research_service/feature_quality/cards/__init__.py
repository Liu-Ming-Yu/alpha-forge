"""Feature-card parsing and feature-audit manifest helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    FeatureDefinition,
    FeatureExpectedSign,
    FeatureProductionState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.core.contracts import ArtifactStore


def load_feature_definition(path: Path) -> FeatureDefinition:
    """Load a feature card JSON file into a FeatureDefinition."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return feature_definition_from_payload(raw)


def feature_definition_from_payload(raw: Mapping[str, object]) -> FeatureDefinition:
    """Build a FeatureDefinition from a JSON-like mapping."""
    return FeatureDefinition(
        name=str(raw["name"]),
        version=str(raw["version"]),
        owner=str(raw["owner"]),
        economic_thesis=str(raw["economic_thesis"]),
        source_datasets=_string_tuple(raw.get("source_datasets"), field_name="source_datasets"),
        required_lags=_string_tuple(raw.get("required_lags"), field_name="required_lags"),
        valid_universe=str(raw.get("valid_universe", "unspecified")),
        expected_sign=FeatureExpectedSign(str(raw.get("expected_sign", "positive"))),
        horizon_days=int(str(raw.get("horizon_days", 21))),
        expected_turnover=str(raw.get("expected_turnover", "medium")),
        state=FeatureProductionState(str(raw.get("state", "draft"))),
        failure_modes=_string_tuple(raw.get("failure_modes"), field_name="failure_modes"),
        risk_exposures=_string_tuple(raw.get("risk_exposures"), field_name="risk_exposures"),
    )


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a JSON array")
    return tuple(str(item) for item in value)


def generated_shadow_feature_definition(
    *,
    feature_name: str,
    feature_set_version: str,
    horizon_days: int,
) -> FeatureDefinition:
    """Create a reviewable shadow-only feature card for automatic audits."""
    return FeatureDefinition(
        name=feature_name,
        version=feature_set_version,
        owner="research-campaign",
        economic_thesis=(
            "Automatically generated shadow audit card. Human review is required "
            "before this feature can move to paper or live production state."
        ),
        source_datasets=(f"feature_set:{feature_set_version}",),
        required_lags=("FeatureVector.available_at must be <= decision time",),
        valid_universe="campaign contracts universe",
        expected_sign=FeatureExpectedSign.POSITIVE,
        horizon_days=horizon_days,
        expected_turnover="unknown until audited",
        state=FeatureProductionState.SHADOW,
        failure_modes=("shadow-only auto card; not eligible for live promotion",),
        risk_exposures=("unknown",),
    )


def feature_schema_hash(feature: FeatureDefinition, feature_set_version: str) -> str:
    payload = {
        "feature": feature_to_payload(feature),
        "feature_set_version": feature_set_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_feature_audit_manifest(
    path: Path,
    *,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, object]:
    if artifact_store is not None:
        return dict(artifact_store.read_json(str(path)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"feature audit manifest must be a JSON object: {path}")
    return dict(payload)


def list_feature_audit_manifests(
    root: Path,
    *,
    limit: int = 50,
    artifact_store: ArtifactStore | None = None,
) -> list[dict[str, object]]:
    base = root / "research" / "feature_audits"
    if artifact_store is not None:
        manifest_rows = [
            dict(row)
            for row in artifact_store.list_json(str(base), "*/*/*/feature_audit_manifest.json")
        ]
        manifest_rows.sort(key=lambda row: str(row.get("generated_at", "")), reverse=True)
        return manifest_rows[: max(1, limit)]
    if not base.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in base.glob("*/*/*/feature_audit_manifest.json"):
        try:
            payload = read_feature_audit_manifest(path)
        except (OSError, json.JSONDecodeError):
            continue
        payload["artifact_root"] = str(path.parent)
        rows.append(payload)
    rows.sort(key=lambda row: str(row.get("generated_at", "")), reverse=True)
    return rows[: max(1, limit)]


def feature_to_payload(feature: FeatureDefinition) -> dict[str, object]:
    payload = asdict(feature)
    payload["expected_sign"] = feature.expected_sign.value
    payload["state"] = feature.state.value
    payload["source_datasets"] = list(feature.source_datasets)
    payload["required_lags"] = list(feature.required_lags)
    payload["failure_modes"] = list(feature.failure_modes)
    payload["risk_exposures"] = list(feature.risk_exposures)
    return payload
