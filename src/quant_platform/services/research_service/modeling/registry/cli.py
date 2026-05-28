"""``quant-platform model-registry`` CLI dispatch.

Thin wrapper around :class:`PostgresModelRegistry` that exposes the
promote / retire / list / diff / rollback workflows to operators.
Split from ``__main__.py`` so the argparse wiring stays short and the
registry-specific logic is testable without spinning up argparse.

Retires **R-GOV-02**: the registry is writable from a Postgres
transaction in production, but the rollback flow was previously only
available via direct SQL.  This module standardises the flow and adds
a diff helper that surfaces config/version drift at a glance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog

if TYPE_CHECKING:
    import argparse
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import ModelRegistryRepository

log = structlog.get_logger(__name__)


class RegisteredModelLike(Protocol):
    model_id: uuid.UUID
    strategy_name: str
    model_version: str
    feature_set_version: str
    created_at: datetime
    metadata: dict[str, object]
    active: bool


class ModelRegistryCliRepository(Protocol):
    async def list_models(self) -> list[RegisteredModelLike]: ...

    async def register_model(
        self,
        *,
        strategy_name: str,
        model_version: str,
        feature_set_version: str,
        as_of: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RegisteredModelLike: ...

    async def retire_model(self, strategy_name: str) -> int: ...

    async def get_model(
        self,
        strategy_name: str,
        model_version: str,
    ) -> RegisteredModelLike | None: ...

    async def rollback_to_version(
        self,
        strategy_name: str,
        model_version: str,
    ) -> RegisteredModelLike: ...


def build_model_registry(_dsn: str) -> ModelRegistryRepository:
    """Compatibility injection hook for tests; bootstrap supplies production registries."""
    raise RuntimeError("model registry must be supplied by bootstrap")


async def dispatch(
    *,
    settings: PlatformSettings,
    subcommand: str,
    args: argparse.Namespace,
    registry: ModelRegistryCliRepository | None = None,
) -> None:
    """Route a ``model-registry`` subcommand to its handler.

    The CLI only supports the Postgres-backed registry - the in-memory
    variant is a test double, not an operator tool.  ``SystemExit`` is
    raised when the DSN is absent so operators immediately see the real
    failure mode rather than a confusing in-memory no-op.
    """
    if not settings.storage.postgres_dsn:
        raise SystemExit(
            "model-registry CLI requires QP__STORAGE__POSTGRES_DSN.  "
            "The in-memory registry is a test double."
        )
    active_registry = registry or cast(
        "ModelRegistryCliRepository",
        build_model_registry(settings.storage.postgres_dsn),
    )

    if subcommand == "list":
        await _cmd_list(active_registry)
    elif subcommand == "promote":
        await _cmd_promote(active_registry, args)
    elif subcommand == "retire":
        await _cmd_retire(active_registry, args)
    elif subcommand == "diff":
        await _cmd_diff(active_registry, args)
    elif subcommand == "rollback":
        await _cmd_rollback(active_registry, args)
    else:  # pragma: no cover - argparse rejects earlier
        raise SystemExit(f"unknown model-registry subcommand: {subcommand}")


async def _cmd_list(registry: ModelRegistryCliRepository) -> None:
    models = await registry.list_models()
    if not models:
        print("No registered models.")
        return
    print(f"{'Strategy':<32} {'Version':<16} {'FSV':<10} {'Active':<6} {'Created (timezone.utc)'}")
    for m in models:
        print(
            f"{m.strategy_name:<32} {m.model_version:<16} "
            f"{m.feature_set_version:<10} {('yes' if m.active else 'no'):<6} "
            f"{m.created_at.astimezone(UTC).isoformat()}"
        )


async def _cmd_promote(registry: ModelRegistryCliRepository, args: argparse.Namespace) -> None:
    config = _load_config(args.config_path) if getattr(args, "config_path", None) else {}
    metadata = _load_config(args.metadata_path) if getattr(args, "metadata_path", None) else {}
    feature_set_version = getattr(args, "feature_set_version", None) or args.engine_version
    metadata.update({"config": config, "engine_version": args.engine_version})
    artifact_manifest = getattr(args, "artifact_manifest", None)
    if artifact_manifest:
        metadata["artifact_manifest"] = str(artifact_manifest)
        metadata["boosting"] = _load_boosting_manifest_metadata(artifact_manifest)
    registered = await registry.register_model(
        strategy_name=args.name,
        model_version=args.version,
        feature_set_version=feature_set_version,
        as_of=datetime.now(tz=UTC),
        metadata=metadata,
    )
    log.info(
        "model_registry.promote.complete",
        strategy=registered.strategy_name,
        version=registered.model_version,
        model_id=str(registered.model_id),
    )


async def _cmd_retire(registry: ModelRegistryCliRepository, args: argparse.Namespace) -> None:
    retired = await registry.retire_model(args.name)
    if retired == 0:
        raise SystemExit(f"no active model found for strategy={args.name!r}")
    log.info("model_registry.retire.complete", strategy=args.name, rows=retired)


async def _cmd_diff(registry: ModelRegistryCliRepository, args: argparse.Namespace) -> None:
    source = await registry.get_model(args.name, args.from_version)
    target = await registry.get_model(args.name, args.to_version)
    if source is None or target is None:
        missing = []
        if source is None:
            missing.append(args.from_version)
        if target is None:
            missing.append(args.to_version)
        raise SystemExit(f"model versions not found for strategy={args.name!r}: {missing}")
    src_payload = _serialize_for_diff(source)
    tgt_payload = _serialize_for_diff(target)
    diff = unified_diff(
        src_payload.splitlines(keepends=True),
        tgt_payload.splitlines(keepends=True),
        fromfile=f"{args.name}@{args.from_version}",
        tofile=f"{args.name}@{args.to_version}",
    )
    print("".join(diff) or "(identical)")


async def _cmd_rollback(registry: ModelRegistryCliRepository, args: argparse.Namespace) -> None:
    try:
        restored = await registry.rollback_to_version(args.name, args.to_version)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc
    log.info(
        "model_registry.rollback.complete",
        strategy=restored.strategy_name,
        version=restored.model_version,
        model_id=str(restored.model_id),
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"config payload not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"config payload must be a JSON object: {path}")
    return payload


def _load_boosting_manifest_metadata(path: Path) -> dict[str, Any]:
    manifest = _load_config(path)
    metadata: dict[str, Any] = {
        "model_type": manifest.get("model_type"),
        "model_version": manifest.get("model_version"),
        "feature_set_version": manifest.get("feature_set_version"),
        "feature_schema_hash": manifest.get("feature_schema_hash"),
        "xgboost_version": manifest.get("xgboost_version"),
        "device": manifest.get("device"),
        "objective": manifest.get("objective"),
        "metrics": manifest.get("metrics", {}),
    }
    metrics_path = manifest.get("metrics_path")
    if metrics_path:
        metrics_file = Path(metrics_path)
        if not metrics_file.is_absolute():
            metrics_file = path.parent / metrics_file
        if metrics_file.is_file():
            metadata["metrics"] = _load_config(metrics_file)
    return metadata


def _serialize_for_diff(model: RegisteredModelLike) -> str:
    return json.dumps(
        {
            "strategy_name": model.strategy_name,
            "model_version": model.model_version,
            "feature_set_version": model.feature_set_version,
            "active": model.active,
            "metadata": model.metadata,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )
