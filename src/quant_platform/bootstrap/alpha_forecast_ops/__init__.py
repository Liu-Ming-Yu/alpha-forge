"""Materialize current promoted-source forecast evidence."""

from __future__ import annotations

from quant_platform.bootstrap.alpha_forecast_ops.cli import alpha_materialize_forecasts_command
from quant_platform.bootstrap.alpha_forecast_ops.materialize import materialize_alpha_forecasts

__all__ = [
    "alpha_materialize_forecasts_command",
    "materialize_alpha_forecasts",
]
