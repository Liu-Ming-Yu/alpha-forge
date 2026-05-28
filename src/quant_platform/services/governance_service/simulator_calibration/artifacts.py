"""Simulator calibration artifact serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.services.governance_service.simulator_calibration.models import (
        CalibrationReport,
    )


def calibration_payload(report: CalibrationReport) -> dict[str, object]:
    """Return JSON-safe payload for ``CalibrationReport``."""
    return {
        "generated_at": report.generated_at.isoformat(),
        "sample_count": report.sample_count,
        "insufficient_data": report.insufficient_data,
        "floor_bps": report.floor_bps,
        "overall": {
            "median_bps": report.overall_median_bps,
            "p90_bps": report.overall_p90_bps,
            "recommended_bps": report.overall_recommended_bps,
        },
        "buckets": [asdict(b) for b in report.buckets],
    }


def write_calibration_report(report: CalibrationReport, output: Path) -> Path:
    """Write the report JSON to disk. Returns the canonical path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(calibration_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
