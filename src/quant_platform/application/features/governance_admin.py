"""Administrative feature-governance use case helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.application.features.governance_payloads import (
    feature_audit_result_payload,
    feature_state_meets_minimum,
)
from quant_platform.application.features.governance_requests import (
    FeatureAuditAssertRequest,
    FeatureAuditCommandResult,
    FeatureAuditRetireRequest,
    FeatureAuditStatusRequest,
)
from quant_platform.application.operator.payload_coercion import optional_mapping, optional_sequence
from quant_platform.core.domain.research import (
    FeatureAuditResult,
    FeatureDefinition,
    FeatureExpectedSign,
    FeatureProductionState,
)
from quant_platform.services.research_service.feature_quality.audit.feature_audit import (
    feature_schema_hash,
    list_feature_audit_manifests,
    read_feature_audit_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.core.contracts import ArtifactStore, FeatureAuditRepository


async def feature_audit_status(
    *,
    request: FeatureAuditStatusRequest,
    object_store_root: Path,
    repository: FeatureAuditRepository | None,
    artifact_store: ArtifactStore | None,
) -> FeatureAuditCommandResult:
    """Return latest feature audit state from Postgres or artifact manifests."""
    if repository is not None:
        if request.feature_name and request.feature_version:
            latest = await repository.latest_feature_audit(
                request.feature_name,
                request.feature_version,
            )
            audit_rows = [latest] if latest is not None else []
        else:
            audit_rows = await repository.list_feature_audits(
                feature_name=request.feature_name,
                limit=request.limit,
            )
        audits = [feature_audit_result_payload(row) for row in audit_rows]
        return FeatureAuditCommandResult({"audits": audits, "count": len(audits)})

    root = request.output_root or object_store_root
    manifest_rows = list_feature_audit_manifests(
        root,
        limit=request.limit,
        artifact_store=artifact_store,
    )
    if request.feature_name:
        manifest_rows = [
            row
            for row in manifest_rows
            if optional_mapping(row.get("feature"), name="feature").get("name")
            == request.feature_name
        ]
    if request.feature_version:
        manifest_rows = [
            row
            for row in manifest_rows
            if optional_mapping(row.get("feature"), name="feature").get("version")
            == request.feature_version
        ]
    return FeatureAuditCommandResult({"audits": manifest_rows, "count": len(manifest_rows)})


async def assert_feature_audit(
    *,
    request: FeatureAuditAssertRequest,
    repository: FeatureAuditRepository | None,
    artifact_store: ArtifactStore | None,
) -> FeatureAuditCommandResult:
    """Enforce the latest feature audit promotion gate."""
    if request.manifest is not None:
        payload = read_feature_audit_manifest(
            request.manifest,
            artifact_store=artifact_store,
        )
        passed = bool(payload.get("passed"))
        state = str(optional_mapping(payload.get("feature"), name="feature").get("state", "draft"))
        blockers = list(optional_sequence(payload.get("blockers"), name="blockers"))
        out = dict(payload)
    else:
        if not request.feature_name:
            raise ValueError("features audit assert requires --manifest or --feature-name")
        if repository is None:
            raise ValueError("repository-backed feature audit assert requires Postgres")
        result = await repository.latest_feature_audit(
            request.feature_name,
            request.feature_version,
        )
        if result is None:
            raise ValueError(f"no feature audit found for {request.feature_name}")
        passed = result.passed
        state = result.status.value
        blockers = list(result.blockers)
        out = feature_audit_result_payload(result)

    if not feature_state_meets_minimum(state, request.minimum_state):
        blockers.append(f"state {state!r} is below required {request.minimum_state!r}")
        passed = False
        out["blockers"] = blockers
    return FeatureAuditCommandResult(payload=out, passed=passed)


async def retire_feature_audit(
    *,
    request: FeatureAuditRetireRequest,
    repository: FeatureAuditRepository | None,
) -> FeatureAuditCommandResult:
    """Persist a retired marker for a feature/version."""
    if repository is None:
        raise ValueError("features audit retire requires QP__STORAGE__POSTGRES_DSN")
    now = datetime.now(tz=UTC)
    feature = FeatureDefinition(
        name=request.feature_name,
        version=request.feature_version,
        owner="operator",
        economic_thesis="Retired by operator through feature audit governance CLI.",
        source_datasets=("retirement-marker",),
        required_lags=("not applicable",),
        valid_universe="not applicable",
        expected_sign=FeatureExpectedSign.NON_MONOTONIC,
        horizon_days=1,
        expected_turnover="not applicable",
        state=FeatureProductionState.RETIRED,
        failure_modes=("retired",),
    )

    result = FeatureAuditResult(
        audit_id=uuid.uuid4(),
        feature_name=request.feature_name,
        feature_version=request.feature_version,
        feature_set_version=request.feature_set_version,
        as_of=now,
        sample_start=now,
        sample_end=now,
        status=FeatureProductionState.RETIRED,
        passed=False,
        metrics={},
        gate_results={"retired": False},
        artifact_uri="",
        schema_hash=feature_schema_hash(feature, request.feature_set_version),
        code_commit="operator-retired",
        blockers=(request.reason,),
    )
    await repository.save_feature_audit(result)
    return FeatureAuditCommandResult(feature_audit_result_payload(result), passed=False)


__all__ = ["assert_feature_audit", "feature_audit_status", "retire_feature_audit"]
