"""Compatibility exports for research campaign metric helpers."""

from __future__ import annotations

from quant_platform.services.research_service.campaigns.evaluation.slippage_calibration import (
    calibrated_slippage_bps_per_turnover,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    attribution_by_metadata as _attribution_by_metadata,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    bootstrap_ic_ci as _bootstrap_ic_ci,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    daily_metrics as _daily_metrics,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    feature_stability as _feature_stability,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    fit_correlation_weights as _fit_correlation_weights,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    score_features as _score,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    top_minus_bottom_decile_ic as _top_minus_bottom_decile_ic,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    compound_return as _compound_return,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    equity_curve as _equity_curve,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    max_drawdown as _max_drawdown,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    sharpe as _sharpe,
)
from quant_platform.services.research_service.sampling.eligibility import (
    eligibility as _eligibility,
)

__all__ = [
    "_attribution_by_metadata",
    "_bootstrap_ic_ci",
    "_compound_return",
    "_daily_metrics",
    "_eligibility",
    "_equity_curve",
    "_feature_stability",
    "_fit_correlation_weights",
    "_max_drawdown",
    "_score",
    "_sharpe",
    "_top_minus_bottom_decile_ic",
    "calibrated_slippage_bps_per_turnover",
]
