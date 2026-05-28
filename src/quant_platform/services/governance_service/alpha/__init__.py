"""Governed alpha promotion/readiness helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.application.features.admission import (
    ordered_feature_schema_hash as _ordered_feature_schema_hash,
)
from quant_platform.core.domain.production import RuntimeHeartbeat
from quant_platform.services.governance_service.alpha.alpha_manifest_checks import (
    text_manifest_checks,
)
from quant_platform.services.governance_service.gates.signal_gate import signal_gate_status

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        ModelRegistryRepository,
        OperationalReadinessRepository,
        SignalPromotionGate,
    )


def build_model_registry(_dsn: str) -> ModelRegistryRepository:
    """Compatibility injection hook for tests; bootstrap supplies production registries."""
    raise RuntimeError("model registry must be supplied by bootstrap")


def build_performance_repository(_dsn: str | None) -> OperationalReadinessRepository:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("performance repository must be supplied by bootstrap")


ordered_feature_schema_hash = _ordered_feature_schema_hash


async def alpha_assert(
    settings: PlatformSettings,
    *,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    artifact_manifest: Path | None = None,
    signal_gate: SignalPromotionGate | None = None,
    model_registry: ModelRegistryRepository | None = None,
) -> dict[str, object]:
    """Return production-readiness checks for one promoted alpha source."""
    checks: list[dict[str, object]] = []

    gate = await signal_gate_status(
        settings,
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=as_of,
        gate=signal_gate,
    )
    checks.append(
        {
            "name": "signal_gate_passed",
            "passed": gate.passed,
            "detail": (
                f"state={gate.state.value} observations={gate.observations} "
                f"rolling_ic={gate.rolling_ic:.4f}"
            ),
        }
    )

    active_model = None
    if settings.storage.postgres_dsn:
        registry = model_registry or build_model_registry(settings.storage.postgres_dsn)
        active_model = await registry.get_active_model(signal_name)
    checks.append(
        {
            "name": "active_model_registered",
            "passed": active_model is not None,
            "detail": active_model.model_version if active_model is not None else "no active model",
        }
    )

    manifest_payload: dict[str, Any] = {}
    if artifact_manifest is not None and artifact_manifest.is_file():
        manifest_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    checks.append(
        {
            "name": "artifact_manifest_present",
            "passed": bool(manifest_payload),
            "detail": str(artifact_manifest)
            if artifact_manifest
            else "no artifact manifest supplied",
        }
    )

    if signal_type == "xgboost" and manifest_payload:
        metrics = dict(manifest_payload.get("metrics", {}))
        checks.extend(
            [
                {
                    "name": "xgboost_validation_ic",
                    "passed": float(metrics.get("validation_ic", 0.0))
                    >= settings.production.text_gate_min_ic,
                    "detail": str(metrics.get("validation_ic")),
                },
                {
                    "name": "xgboost_training_groups",
                    "passed": int(metrics.get("train_groups", 0)) >= 252,
                    "detail": str(metrics.get("train_groups")),
                },
                {
                    "name": "xgboost_validation_groups",
                    "passed": int(metrics.get("validation_groups", 0)) >= 20,
                    "detail": str(metrics.get("validation_groups")),
                },
                {
                    "name": "xgboost_feature_coverage",
                    "passed": float(metrics.get("feature_coverage", 1.0)) >= 0.95,
                    "detail": str(metrics.get("feature_coverage", 1.0)),
                },
            ]
        )
        booster_path = Path(str(manifest_payload.get("booster_path", "")))
        if not booster_path.is_absolute() and artifact_manifest is not None:
            booster_path = artifact_manifest.parent / booster_path
        expected_hash = manifest_payload.get("booster_sha256")
        actual_hash = _sha256_file(booster_path) if booster_path.is_file() else ""
        checks.append(
            {
                "name": "artifact_hash_verified",
                "passed": bool(expected_hash) and expected_hash == actual_hash,
                "detail": actual_hash or "booster missing or hash absent",
            }
        )
    if signal_type == "text" and manifest_payload:
        checks.extend(
            text_manifest_checks(
                settings,
                manifest_payload=manifest_payload,
                active_feature_set_version=active_model.feature_set_version
                if active_model is not None
                else "",
            )
        )

    passed = all(bool(check["passed"]) for check in checks)
    return {
        "signal_name": signal_name,
        "signal_type": signal_type,
        "as_of": as_of.isoformat(),
        "passed": passed,
        "checks": checks,
    }


async def alpha_promote(
    settings: PlatformSettings,
    *,
    signal_name: str,
    signal_type: str,
    model_version: str,
    feature_set_version: str,
    engine_version: str,
    artifact_manifest: Path | None,
    rollback_target: str,
    as_of: datetime,
    model_registry: ModelRegistryRepository | None = None,
    heartbeat_repository: OperationalReadinessRepository | None = None,
) -> dict[str, object]:
    if not settings.storage.postgres_dsn:
        raise RuntimeError("alpha promote requires QP__STORAGE__POSTGRES_DSN")
    registry = model_registry or build_model_registry(settings.storage.postgres_dsn)
    metadata: dict[str, object] = {
        "alpha": {
            "signal_type": signal_type,
            "artifact_manifest": str(artifact_manifest) if artifact_manifest else "",
            "approved_source_weights": settings.alpha.source_weights,
            "rollback_target": rollback_target,
            "ramp_level": str(settings.alpha.live_ramp_initial),
        }
    }
    model = await registry.register_model(
        strategy_name=signal_name,
        model_version=model_version,
        feature_set_version=feature_set_version,
        as_of=as_of,
        metadata={**metadata, "engine_version": engine_version},
    )
    repository = heartbeat_repository or build_performance_repository(settings.storage.postgres_dsn)
    await repository.save_runtime_heartbeat(
        RuntimeHeartbeat(
            component=f"alpha:{signal_type}:{signal_name}",
            as_of=as_of,
            status="ok",
            detail=f"promoted {model_version} rollback_target={rollback_target}",
        )
    )
    return {"promoted": True, "model_id": str(model.model_id), "model_version": model.model_version}


async def alpha_rollback(
    settings: PlatformSettings,
    *,
    signal_name: str,
    target_version: str,
    as_of: datetime,
    model_registry: ModelRegistryRepository | None = None,
    heartbeat_repository: OperationalReadinessRepository | None = None,
) -> dict[str, object]:
    if not settings.storage.postgres_dsn:
        raise RuntimeError("alpha rollback requires QP__STORAGE__POSTGRES_DSN")
    registry = model_registry or build_model_registry(settings.storage.postgres_dsn)
    model = await registry.rollback_to_version(signal_name, target_version)
    repository = heartbeat_repository or build_performance_repository(settings.storage.postgres_dsn)
    await repository.save_runtime_heartbeat(
        RuntimeHeartbeat(
            component=f"alpha:rollback:{signal_name}",
            as_of=as_of,
            status="ok",
            detail=f"rolled back to {target_version}; set ensemble to classical-only until restart",
        )
    )
    return {
        "rolled_back": True,
        "model_id": str(model.model_id),
        "model_version": model.model_version,
    }


def alpha_ramp(settings: PlatformSettings, *, clean_live_days: int) -> dict[str, object]:
    if clean_live_days >= 60:
        level = settings.alpha.live_ramp_after_60d
    elif clean_live_days >= 20:
        level = settings.alpha.live_ramp_after_20d
    else:
        level = settings.alpha.live_ramp_initial
    return {
        "clean_live_days": clean_live_days,
        "ramp_level": str(level),
        "max_non_classical_weight": str(min(level, settings.alpha.live_ramp_after_60d)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
