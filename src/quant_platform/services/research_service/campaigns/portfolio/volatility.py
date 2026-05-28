"""Training-window volatility scaling for campaign portfolios."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_platform.core.constants import TRADING_DAYS_PER_YEAR
from quant_platform.services.research_service.campaigns.portfolio.evaluation import (
    evaluate_long_only_portfolio,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
    FoldVolatilityScale,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def fit_fold_volatility_scale(
    train_scored: Sequence[tuple[SupervisedAlphaSample, float]],
    *,
    config: CampaignPortfolioConfig,
) -> FoldVolatilityScale:
    """Estimate fold exposure from training data only, never test returns."""
    train_eval = evaluate_long_only_portfolio(
        train_scored,
        slippage_bps_per_turnover=0.0,
        config=config,
        previous_weights=None,
        exposure_scale=1.0,
    )
    lookback_returns = train_eval.daily_returns[-int(config.vol_lookback_days) :]
    realized_vol = _annualized_volatility(lookback_returns)
    effective_vol = max(realized_vol, float(config.vol_floor))
    raw_scale = float(config.vol_target) / effective_vol
    exposure_scale = min(1.0, raw_scale)
    return FoldVolatilityScale(
        exposure_scale=exposure_scale,
        train_realized_vol=realized_vol,
        train_effective_vol=effective_vol,
        raw_vol_scale=raw_scale,
        train_observations=len(lookback_returns),
    )


def _annualized_volatility(returns: Sequence[float]) -> float:
    values = [float(value) for value in returns if math.isfinite(float(value))]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance)) * math.sqrt(TRADING_DAYS_PER_YEAR)


__all__ = ["fit_fold_volatility_scale"]
