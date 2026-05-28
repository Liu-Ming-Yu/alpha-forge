"""Support helpers for governed text-event bootstrap commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.research.common import _json_default
from quant_platform.services.research_service.text.features.client import (
    llm_api_key_env_name,
    resolve_llm_api_key,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings


def require_text_durable(settings: PlatformSettings) -> None:
    if not settings.storage.postgres_dsn:
        raise OperatorUsageError(
            "text-events requires QP__STORAGE__POSTGRES_DSN so text events and "
            "feature vectors persist across commands"
        )


def missing_cik_symbols(
    contracts: dict[uuid.UUID, dict[str, object]],
    cik_map: dict[str, str],
) -> list[str]:
    missing: list[str] = []
    for contract in contracts.values():
        symbol = str(contract.get("symbol", "")).upper().strip()
        if symbol and symbol not in cik_map:
            missing.append(symbol)
    return sorted(missing)


def llm_credential_blocker(settings: PlatformSettings) -> str:
    if resolve_llm_api_key(settings.llm.provider):
        return ""
    return f"{llm_api_key_env_name(settings.llm.provider)} is required"


def text_artifact_root(settings: PlatformSettings) -> Path:
    return Path(settings.storage.object_store_root) / "research" / "text_events"


def text_event_slug(prefix: str, start: datetime, end: datetime) -> str:
    return f"{prefix}_{start.astimezone(UTC):%Y-%m-%d}_{end.astimezone(UTC):%Y-%m-%d}"


def write_text_manifest(
    settings: PlatformSettings,
    slug: str,
    filename: str,
    payload: dict[str, object],
) -> Path:
    root = text_artifact_root(settings) / slug
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def blocked_text_payload(
    settings: PlatformSettings,
    slug: str,
    reason: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "passed": False,
        "reason": reason,
        "details": details or {},
    }
    root = text_artifact_root(settings) / "_blocked" / slug
    root.mkdir(parents=True, exist_ok=True)
    path = root / "blocked_text_source_summary.json"
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["blocked_text_source_summary"] = str(path)
    return payload


__all__ = [
    "blocked_text_payload",
    "llm_credential_blocker",
    "missing_cik_symbols",
    "require_text_durable",
    "text_artifact_root",
    "text_event_slug",
    "write_text_manifest",
]
