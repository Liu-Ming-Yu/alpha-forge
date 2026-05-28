"""Low-level helpers for live-LLM startup governance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import ProductionProfile
from quant_platform.core.domain.research import FeatureProductionState

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings


def _parse_timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise RuntimeError(f"timestamp must be ISO-8601: {raw!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"timestamp must be timezone-aware: {raw!r}")
    return parsed.astimezone(UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_minimum_state(profile: ProductionProfile) -> FeatureProductionState:
    if profile == ProductionProfile.LLM_LIVE_REHEARSAL:
        return FeatureProductionState.PAPER
    return FeatureProductionState.LIVE


def _source_weights_payload(settings: PlatformSettings) -> dict[str, float]:
    return {
        str(name): float(weight) for name, weight in sorted(settings.alpha.source_weights.items())
    }


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _object_mapping(value: object) -> MappingABC[object, object]:
    if isinstance(value, MappingABC):
        return value
    return {}


def _list_payload(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _coerce_float(value: object) -> float:
    return float(str(value))
