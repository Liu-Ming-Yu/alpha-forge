"""Runtime feature-admission policy.

This policy is intentionally independent of CLI and repository construction.
Bootstrap code supplies the audited feature rows; runtime model loaders can use
the policy before permitting paper/live scoring.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from quant_platform.application.operator.payload_coercion import (
    optional_mapping,
    optional_sequence,
    require_float,
)
from quant_platform.core.domain.research import (
    FeatureAuditResult,
    FeatureProductionState,
    FeatureVector,
)

if TYPE_CHECKING:
    from pathlib import Path

_STATE_ORDER = {
    FeatureProductionState.DRAFT.value: 0,
    FeatureProductionState.SHADOW.value: 1,
    FeatureProductionState.PAPER.value: 2,
    FeatureProductionState.LIVE.value: 3,
    FeatureProductionState.RETIRED.value: -1,
}


@dataclass(frozen=True)
class FeatureAdmissionDecision:
    passed: bool
    blockers: tuple[str, ...]
    feature_schema_hash: str


class FeatureAdmissionPolicy:
    """Fail-closed admission checks for production feature sets."""

    def __init__(self, audits: Sequence[FeatureAuditResult]) -> None:
        self._latest: dict[tuple[str, str], FeatureAuditResult] = {}
        for audit in sorted(audits, key=lambda row: row.as_of):
            self._latest[(audit.feature_name, audit.feature_version)] = audit

    def evaluate(
        self,
        *,
        feature_names: Sequence[str],
        feature_versions: Mapping[str, str],
        feature_set_version: str,
        minimum_state: FeatureProductionState,
        model_feature_schema_hash: str,
    ) -> FeatureAdmissionDecision:
        ordered_names = tuple(str(name) for name in feature_names)
        schema_hash = ordered_feature_schema_hash(ordered_names)
        blockers: list[str] = []
        if schema_hash != model_feature_schema_hash:
            blockers.append(
                "model feature_schema_hash does not match admitted ordered feature names"
            )
        for name in ordered_names:
            version = feature_versions.get(name)
            if not version:
                blockers.append(f"missing feature version for {name!r}")
                continue
            audit = self._latest.get((name, version))
            if audit is None:
                blockers.append(f"missing feature audit for {name!r} version {version!r}")
                continue
            blockers.extend(
                _audit_blockers(
                    audit,
                    feature_set_version=feature_set_version,
                    minimum_state=minimum_state,
                )
            )
        return FeatureAdmissionDecision(
            passed=not blockers,
            blockers=tuple(blockers),
            feature_schema_hash=schema_hash,
        )

    def assert_admitted(
        self,
        *,
        feature_names: Sequence[str],
        feature_versions: Mapping[str, str],
        feature_set_version: str,
        minimum_state: FeatureProductionState,
        model_feature_schema_hash: str,
    ) -> None:
        decision = self.evaluate(
            feature_names=feature_names,
            feature_versions=feature_versions,
            feature_set_version=feature_set_version,
            minimum_state=minimum_state,
            model_feature_schema_hash=model_feature_schema_hash,
        )
        if not decision.passed:
            raise RuntimeError("; ".join(decision.blockers))

    def assert_vectors_admitted(
        self,
        *,
        vectors: Sequence[FeatureVector],
        feature_versions: Mapping[str, str],
        minimum_state: FeatureProductionState,
        model_feature_schema_hash: str,
    ) -> None:
        if not vectors:
            raise RuntimeError("cannot admit an empty feature vector batch")
        feature_set_version = vectors[0].feature_set_version
        names = sorted({name for vector in vectors for name in vector.features})
        if any(vector.feature_set_version != feature_set_version for vector in vectors):
            raise RuntimeError("feature vector batch mixes feature_set_version values")
        self.assert_admitted(
            feature_names=names,
            feature_versions=feature_versions,
            feature_set_version=feature_set_version,
            minimum_state=minimum_state,
            model_feature_schema_hash=model_feature_schema_hash,
        )


def ordered_feature_schema_hash(feature_names: Sequence[str]) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_feature_artifacts_admitted(
    *,
    audit_root: Path,
    feature_names: Sequence[str],
    feature_versions: Mapping[str, str],
    feature_set_version: str,
    minimum_state: FeatureProductionState,
    model_feature_schema_hash: str,
) -> FeatureAdmissionDecision:
    """Load local feature-audit evidence and fail closed for production use.

    The durable Postgres repository remains the system of record when available.
    This artifact reader is the synchronous runtime bridge used by session
    builders while durable repository checks stay in async bootstrap paths.
    """

    audits = load_feature_audit_results_from_artifacts(
        audit_root,
        feature_names=feature_names,
    )
    if not audits:
        raise RuntimeError(
            f"no feature audit manifests found under {audit_root / 'research' / 'feature_audits'}"
        )
    policy = FeatureAdmissionPolicy(audits)
    decision = policy.evaluate(
        feature_names=feature_names,
        feature_versions=feature_versions,
        feature_set_version=feature_set_version,
        minimum_state=minimum_state,
        model_feature_schema_hash=model_feature_schema_hash,
    )
    if not decision.passed:
        raise RuntimeError("; ".join(decision.blockers))
    return decision


def load_feature_audit_results_from_artifacts(
    audit_root: Path,
    *,
    feature_names: Sequence[str] | None = None,
    limit: int = 500,
) -> list[FeatureAuditResult]:
    """Read feature audit manifests from the research artifact tree."""

    base = audit_root / "research" / "feature_audits"
    if not base.exists():
        return []
    wanted = {str(name) for name in feature_names or ()}
    rows: list[FeatureAuditResult] = []
    for path in base.glob("*/*/*/feature_audit_manifest.json"):
        if wanted and not _manifest_path_matches(path, base=base, feature_names=wanted):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_manifest_payload_to_result(payload, path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid feature audit manifest {path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot read feature audit manifest {path}: {exc}") from exc
    rows.sort(key=lambda row: row.as_of, reverse=True)
    return rows[: max(1, limit)]


def _audit_blockers(
    audit: FeatureAuditResult,
    *,
    feature_set_version: str,
    minimum_state: FeatureProductionState,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if audit.feature_set_version != feature_set_version:
        blockers.append(
            f"{audit.feature_name!r} audit feature_set_version "
            f"{audit.feature_set_version!r} != {feature_set_version!r}"
        )
    if not audit.passed:
        blockers.append(f"{audit.feature_name!r} latest audit is not passing")
    if audit.blockers:
        blockers.append(f"{audit.feature_name!r} latest audit has blockers")
    if not _state_at_least(audit.status, minimum_state):
        blockers.append(
            f"{audit.feature_name!r} state {audit.status.value!r} is below {minimum_state.value!r}"
        )
    return tuple(blockers)


def _state_at_least(
    state: FeatureProductionState,
    minimum: FeatureProductionState,
) -> bool:
    return _STATE_ORDER[state.value] >= _STATE_ORDER[minimum.value]


def _manifest_path_matches(
    path: Path,
    *,
    base: Path,
    feature_names: set[str],
) -> bool:
    try:
        rel = path.relative_to(base)
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in feature_names


def _manifest_payload_to_result(
    payload: Mapping[str, object],
    path: Path,
) -> FeatureAuditResult:
    feature_payload = payload["feature"]
    if not isinstance(feature_payload, Mapping):
        raise ValueError(f"feature audit manifest {path} has invalid feature payload")
    feature = dict(feature_payload)
    status = FeatureProductionState(str(feature.get("state", FeatureProductionState.DRAFT.value)))
    passed = bool(payload["passed"])
    if passed and status == FeatureProductionState.DRAFT:
        status = FeatureProductionState.SHADOW
    return FeatureAuditResult(
        audit_id=UUID(str(payload["audit_id"])),
        feature_name=str(feature["name"]),
        feature_version=str(feature["version"]),
        feature_set_version=str(payload["feature_set_version"]),
        as_of=_datetime(str(payload["generated_at"])),
        sample_start=_datetime(str(payload["sample_start"])),
        sample_end=_datetime(str(payload["sample_end"])),
        status=status,
        passed=passed,
        metrics={
            str(key): require_float(value, name=f"metrics.{key}")
            for key, value in optional_mapping(payload.get("metrics"), name="metrics").items()
        },
        gate_results={
            str(key): bool(value)
            for key, value in optional_mapping(
                payload.get("gate_results"),
                name="gate_results",
            ).items()
        },
        artifact_uri=str(path.parent),
        schema_hash=str(payload["schema_hash"]),
        code_commit=str(payload.get("code_commit", "")),
        blockers=tuple(
            str(item) for item in optional_sequence(payload.get("blockers"), name="blockers")
        ),
    )


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {value!r}")
    return parsed
