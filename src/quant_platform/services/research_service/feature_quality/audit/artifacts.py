"""Feature-audit manifest DTOs and artifact writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    FeatureAuditResult,
    FeatureDefinition,
    FeatureProductionState,
)
from quant_platform.services.research_service.feature_quality.cards import (
    feature_to_payload as _feature_to_payload,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.core.contracts import ArtifactStore


@dataclass(frozen=True)
class FeatureAuditManifest:
    """Machine-readable manifest linking all feature audit sidecars."""

    audit_id: uuid.UUID
    feature: FeatureDefinition
    feature_set_version: str
    generated_at: datetime
    sample_start: datetime
    sample_end: datetime
    sample_count: int
    passed: bool
    gate_results: Mapping[str, bool]
    metrics: Mapping[str, float]
    blockers: tuple[str, ...]
    artifacts: Mapping[str, str]
    schema_hash: str
    code_commit: str

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe manifest payload."""
        return {
            "audit_id": str(self.audit_id),
            "feature": _feature_to_payload(self.feature),
            "feature_set_version": self.feature_set_version,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "sample_start": self.sample_start.astimezone(UTC).isoformat(),
            "sample_end": self.sample_end.astimezone(UTC).isoformat(),
            "sample_count": self.sample_count,
            "passed": self.passed,
            "gate_results": dict(self.gate_results),
            "metrics": dict(self.metrics),
            "blockers": list(self.blockers),
            "artifacts": dict(self.artifacts),
            "schema_hash": self.schema_hash,
            "code_commit": self.code_commit,
        }

    def to_result(self, artifact_uri: str) -> FeatureAuditResult:
        status = self.feature.state
        if self.passed and status == FeatureProductionState.DRAFT:
            status = FeatureProductionState.SHADOW
        return FeatureAuditResult(
            audit_id=self.audit_id,
            feature_name=self.feature.name,
            feature_version=self.feature.version,
            feature_set_version=self.feature_set_version,
            as_of=self.generated_at,
            sample_start=self.sample_start,
            sample_end=self.sample_end,
            status=status,
            passed=self.passed,
            metrics=self.metrics,
            gate_results=self.gate_results,
            artifact_uri=artifact_uri,
            schema_hash=self.schema_hash,
            code_commit=self.code_commit,
            blockers=self.blockers,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True)


def feature_audit_artifact_names() -> dict[str, str]:
    """Return the stable feature-audit sidecar filenames."""
    return {
        "feature_card": "feature_card.json",
        "noise_report": "noise_report.json",
        "leakage_report": "leakage_report.json",
        "ic_stability": "ic_stability.json",
        "economic_logic": "economic_logic.json",
        "cost_report": "cost_report.json",
        "incremental_report": "incremental_report.json",
        "manifest": "feature_audit_manifest.json",
    }


def feature_audit_artifact_dir(
    output_root: Path,
    feature: FeatureDefinition,
    audit_id: uuid.UUID,
) -> Path:
    """Return the immutable audit artifact directory for one run."""
    return (
        output_root / "research" / "feature_audits" / feature.name / feature.version / str(audit_id)
    )


def write_feature_audit_artifacts(
    *,
    output_root: Path,
    manifest: FeatureAuditManifest,
    reports: Mapping[str, Mapping[str, object]],
    artifact_store: ArtifactStore | None = None,
) -> Path:
    """Write the standard feature-audit evidence bundle."""
    audit_dir = feature_audit_artifact_dir(output_root, manifest.feature, manifest.audit_id)
    audit_dir.mkdir(parents=True, exist_ok=False)
    artifacts = dict(manifest.artifacts)
    _write_json(
        audit_dir / artifacts["feature_card"],
        _feature_to_payload(manifest.feature),
        output_root=output_root,
        artifact_store=artifact_store,
    )
    for key, filename in (
        ("noise", "noise_report.json"),
        ("leakage", "leakage_report.json"),
        ("ic_stability", "ic_stability.json"),
        ("economic_logic", "economic_logic.json"),
        ("cost", "cost_report.json"),
        ("incremental", "incremental_report.json"),
    ):
        _write_json(
            audit_dir / filename,
            reports[key],
            output_root=output_root,
            artifact_store=artifact_store,
        )
    _write_json(
        audit_dir / artifacts["manifest"],
        manifest.to_payload(),
        output_root=output_root,
        artifact_store=artifact_store,
    )
    return audit_dir / artifacts["manifest"]


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    output_root: Path,
    artifact_store: ArtifactStore | None = None,
) -> None:
    if artifact_store is not None:
        artifact_store.write_json(str(path.resolve().relative_to(output_root.resolve())), payload)
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
