"""Serialization helpers for feature-governance use cases."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.core.domain.research import FeatureAuditResult, FeatureDefinition


def feature_audit_result_payload(result: FeatureAuditResult) -> dict[str, object]:
    """JSON-safe representation for CLI/API responses."""
    return {
        "audit_id": str(result.audit_id),
        "feature_name": result.feature_name,
        "feature_version": result.feature_version,
        "feature_set_version": result.feature_set_version,
        "as_of": result.as_of.isoformat(),
        "sample_start": result.sample_start.isoformat(),
        "sample_end": result.sample_end.isoformat(),
        "status": result.status.value,
        "passed": result.passed,
        "metrics": dict(result.metrics),
        "gate_results": dict(result.gate_results),
        "artifact_uri": result.artifact_uri,
        "schema_hash": result.schema_hash,
        "code_commit": result.code_commit,
        "blockers": list(result.blockers),
    }


def feature_state_meets_minimum(state: str, minimum: str) -> bool:
    order = {"draft": 0, "shadow": 1, "paper": 2, "live": 3, "retired": -1}
    return order.get(state, -1) >= order.get(minimum, 2)


def manifest_path(
    output_root: Path,
    feature: FeatureDefinition,
    audit_id: uuid.UUID,
) -> Path:
    return (
        output_root
        / "research"
        / "feature_audits"
        / feature.name
        / feature.version
        / str(audit_id)
        / "feature_audit_manifest.json"
    )


def csv_names(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def dumps_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)
