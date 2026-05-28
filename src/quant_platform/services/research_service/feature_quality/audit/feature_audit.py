"""Feature-level admission and monitoring orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.services.research_service.feature_quality.audit.artifacts import (
    FeatureAuditManifest,
    feature_audit_artifact_names,
    write_feature_audit_artifacts,
)
from quant_platform.services.research_service.feature_quality.audit.calculations import (
    feature_rows as _feature_rows,
)
from quant_platform.services.research_service.feature_quality.audit.gates import (
    FeatureAuditGatesMixin,
    FeatureAuditThresholds,
)
from quant_platform.services.research_service.feature_quality.cards import (
    feature_definition_from_payload,
    feature_schema_hash,
    generated_shadow_feature_definition,
    list_feature_audit_manifests,
    load_feature_definition,
    read_feature_audit_manifest,
)
from quant_platform.services.research_service.sampling.factory import current_git_commit

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.core.contracts import ArtifactStore
    from quant_platform.core.domain.research import FeatureDefinition
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

__all__ = [
    "FeatureAuditManifest",
    "FeatureAuditRunner",
    "FeatureAuditThresholds",
    "feature_definition_from_payload",
    "feature_schema_hash",
    "generated_shadow_feature_definition",
    "list_feature_audit_manifests",
    "load_feature_definition",
    "read_feature_audit_manifest",
]


class FeatureAuditRunner(FeatureAuditGatesMixin):
    """Run six institutional feature gates over supervised samples."""

    def __init__(
        self,
        *,
        thresholds: FeatureAuditThresholds | None = None,
        slippage_bps_per_turnover: float = 10.0,
        baseline_features: Sequence[str] = (),
        rng_seed: int = 17,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._thresholds = thresholds or FeatureAuditThresholds()
        self._slippage_bps = max(0.0, float(slippage_bps_per_turnover))
        self._baseline_features = tuple(str(name) for name in baseline_features)
        self._rng_seed = rng_seed
        self._artifact_store = artifact_store

    def run(
        self,
        *,
        feature: FeatureDefinition,
        samples: Sequence[SupervisedAlphaSample],
        feature_set_version: str,
        output_root: Path,
        as_of: datetime | None = None,
    ) -> FeatureAuditManifest:
        if not samples:
            raise ValueError("feature audit requires at least one supervised sample")
        ordered = tuple(sorted(samples, key=lambda row: (row.as_of, str(row.instrument_id))))
        sample_start = ordered[0].as_of
        sample_end = ordered[-1].as_of
        generated_at = as_of or datetime.now(tz=UTC)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)

        feature_rows = _feature_rows(ordered, feature.name)
        if not feature_rows:
            raise ValueError(f"samples do not contain feature {feature.name!r}")

        noise = self._noise_gate(ordered, feature.name)
        leakage = self._leakage_gate(feature, ordered, feature_rows)
        stability = self._stability_gate(feature, feature_rows)
        economics = self._economic_gate(feature, stability)
        cost = self._cost_gate(feature, feature_rows)
        incremental = self._incremental_gate(feature, ordered)

        reports = {
            "noise": noise,
            "leakage": leakage,
            "ic_stability": stability,
            "economic_logic": economics,
            "cost": cost,
            "incremental": incremental,
        }
        gate_results = {name: bool(report["passed"]) for name, report in reports.items()}
        blockers = tuple(
            f"{name}: {reason}"
            for name, report in reports.items()
            for reason in _blocker_values(report.get("blockers"))
        )
        metrics: dict[str, float] = {}
        for report in reports.values():
            raw_metrics = report.get("metrics", {})
            if not isinstance(raw_metrics, Mapping):
                raise TypeError("feature audit report metrics must be a mapping")
            metrics.update({str(k): _metric_float(v) for k, v in raw_metrics.items()})

        audit_id = uuid.uuid4()
        schema_hash = feature_schema_hash(feature, feature_set_version)
        manifest = FeatureAuditManifest(
            audit_id=audit_id,
            feature=feature,
            feature_set_version=feature_set_version,
            generated_at=generated_at,
            sample_start=sample_start,
            sample_end=sample_end,
            sample_count=len(ordered),
            passed=all(gate_results.values()),
            gate_results=gate_results,
            metrics=metrics,
            blockers=blockers,
            artifacts=feature_audit_artifact_names(),
            schema_hash=schema_hash,
            code_commit=current_git_commit(),
        )
        write_feature_audit_artifacts(
            output_root=output_root,
            manifest=manifest,
            reports=reports,
            artifact_store=self._artifact_store,
        )
        return manifest


def _metric_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"feature audit metric must be numeric, got {type(value).__name__}")


def _blocker_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("feature audit blockers must be a JSON array")
    return tuple(str(item) for item in value)
