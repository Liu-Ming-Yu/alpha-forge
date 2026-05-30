"""Pluggable alpha models for the walk-forward driver.

``AlphaModel`` / ``FittedAlphaModel`` are the swap points; ``LinearICRanker`` is
the behavior-preserving default, ``GradientBoostedRanker`` is the XGBoost
upgrade, and ``GRUSequenceRanker`` is the PyTorch sequence model. Importing this
package does NOT import xgboost or torch — each heavy model lazy-imports its
backend on first ``fit`` and raises a helpful error if the relevant extra
(``ml`` for xgboost, ``dl`` for torch) is missing.
"""

from __future__ import annotations

from quant_platform.services.research_service.campaigns.models.base import (
    AlphaModel,
    FittedAlphaModel,
)
from quant_platform.services.research_service.campaigns.models.gbdt import GradientBoostedRanker
from quant_platform.services.research_service.campaigns.models.linear import LinearICRanker
from quant_platform.services.research_service.campaigns.models.robust_linear import RobustICRanker
from quant_platform.services.research_service.campaigns.models.sequence import GRUSequenceRanker

__all__ = [
    "AlphaModel",
    "FittedAlphaModel",
    "GRUSequenceRanker",
    "GradientBoostedRanker",
    "LinearICRanker",
    "RobustICRanker",
]
