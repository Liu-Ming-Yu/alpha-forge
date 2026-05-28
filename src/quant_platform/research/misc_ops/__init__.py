"""Miscellaneous research operation wiring."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def _factors_calibrate(
    settings: PlatformSettings,
    samples_path: Path,
    output_dir: Path,
    horizon_days: int,
    l2_lambda: float,
    momentum_scale: float,
) -> None:
    """Fit calibrated factor weights from a JSON samples file."""
    del settings
    from quant_platform.services.research_service.features.factors.calibration import (
        CalibrationSample,
        calibrate,
        write_artifact,
    )

    raw = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = [
        CalibrationSample(
            as_of=datetime.fromisoformat(row["as_of"]),
            instrument_id=uuid.UUID(row["instrument_id"]),
            features=row["features"],
            forward_return=float(row["forward_return"]),
        )
        for row in raw
    ]
    alpha_scale = 1.0 - momentum_scale
    weights = calibrate(
        samples,
        horizon_days=horizon_days,
        l2_lambda=l2_lambda,
        momentum_bloc_scale=momentum_scale,
        alpha_bloc_scale=alpha_scale,
    )
    path = write_artifact(weights, output_dir)
    log.info("factors_calibrate.complete", artifact=str(path))


async def _tearsheet(
    settings: PlatformSettings,
    run_id: uuid.UUID,
    root: Path,
) -> None:
    """Render a Markdown tearsheet for a completed backtest run."""
    del settings
    from quant_platform.services.research_service.reports.tearsheet import render_tearsheet

    path = render_tearsheet(run_id=run_id, root=root)
    log.info("tearsheet.complete", path=str(path))
