"""Calibration artifact loading use cases."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def load_calibration_recommendation_bps(
    path: Path | None,
    *,
    max_age_days: float | None = None,
    as_of: datetime | None = None,
) -> tuple[float | None, dict[str, object]]:
    """Read ``overall.recommended_bps`` from a calibration JSON artifact."""
    if path is None or not path.is_file():
        return None, {"path": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"path": str(path), "error": f"unreadable: {exc}"}
    overall = payload.get("overall") or {}
    bps_raw = overall.get("recommended_bps")
    if not isinstance(bps_raw, (int, float)) or bps_raw <= 0:
        return None, {"path": str(path), "error": "no overall.recommended_bps"}
    metadata: dict[str, object] = {
        "path": str(path),
        "sample_count": payload.get("sample_count"),
        "insufficient_data": bool(payload.get("insufficient_data", False)),
        "generated_at": payload.get("generated_at"),
    }
    if metadata["insufficient_data"]:
        metadata["error"] = "insufficient_data"
        return None, metadata
    if max_age_days is not None and as_of is not None:
        generated_at = _parse_generated_at(payload.get("generated_at"), metadata)
        if generated_at is None:
            return None, metadata
        age_days = (as_of.astimezone(UTC) - generated_at.astimezone(UTC)).total_seconds() / 86400.0
        metadata["age_days"] = age_days
        if age_days > max_age_days:
            metadata["error"] = "stale"
            return None, metadata
    return float(bps_raw), metadata


def _parse_generated_at(raw: object, metadata: dict[str, object]) -> datetime | None:
    if not isinstance(raw, str):
        metadata["error"] = "invalid generated_at"
        return None
    try:
        generated_at = datetime.fromisoformat(raw)
    except ValueError:
        metadata["error"] = "invalid generated_at"
        return None
    return generated_at if generated_at.tzinfo else generated_at.replace(tzinfo=UTC)


__all__ = ["load_calibration_recommendation_bps"]
