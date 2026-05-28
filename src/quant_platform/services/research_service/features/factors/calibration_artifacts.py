"""Artifact IO for factor-weight calibration."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.research_service.features.factors.calibration_models import (
    CalibratedWeights,
)

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)


def write_artifact(weights: CalibratedWeights, directory: Path) -> Path:
    """Write ``weights`` as JSON under ``directory`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = weights.as_of.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"calibrated_weights_{stamp}.json"
    path.write_text(weights.to_json(), encoding="utf-8")
    log.info(
        "factor_calibration.artifact_written",
        path=str(path),
        sample_size=weights.sample_size,
        r2_momentum=weights.r_squared_momentum,
        r2_alpha=weights.r_squared_alpha,
    )
    return path


def read_artifact(path: Path) -> CalibratedWeights:
    """Load a previously written ``CalibratedWeights`` artifact."""
    return CalibratedWeights.from_json(path.read_text(encoding="utf-8"))


__all__ = ["read_artifact", "write_artifact"]
