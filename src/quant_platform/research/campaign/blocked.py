"""Blocked research-campaign artifact helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.research.common import _json_default

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def write_blocked_campaign_summary(
    *,
    output_root: Path,
    sample_slug: str,
    reason: str,
    sample_build: Mapping[str, object],
    feature_audits: Sequence[Mapping[str, object]],
    feature_admission: Mapping[str, object],
    date_policy: str,
    feature_diagnostics_path: Path | None = None,
    feature_attribution_path: Path | None = None,
) -> Path:
    """Write a fail-closed campaign summary when admission blocks training."""
    path = output_root / "_blocked" / sample_slug / "blocked_campaign_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": False,
        "reason": reason,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "date_policy": date_policy,
        "feature_direction_diagnostics": str(feature_diagnostics_path)
        if feature_diagnostics_path is not None
        else None,
        "feature_failure_attribution": str(feature_attribution_path)
        if feature_attribution_path is not None
        else None,
        "sample_build": dict(sample_build),
        "feature_audits": [dict(row) for row in feature_audits],
        "feature_admission": dict(feature_admission),
    }
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


__all__ = ["write_blocked_campaign_summary"]
