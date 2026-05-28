"""Cross-sectional factor specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.services.research_service.features.factors import (
    distance_to_52w_high,
    low_volatility,
    mean_reversion,
    momentum_return,
    momentum_skip1m,
    realized_vol,
    short_term_reversal,
    trend_quality,
    vol_compression_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class FactorSpec:
    """Specification for one alpha factor in the feature pipeline."""

    name: str
    compute: Callable[[Sequence[float]], float]
    is_alpha: bool = True
    winsorize_lower: float = 0.05
    winsorize_upper: float = 0.05


STANDARD_FACTOR_SPECS: list[FactorSpec] = [
    FactorSpec(
        name="momentum_1m",
        compute=lambda c: momentum_return(c, 21),
        is_alpha=True,
    ),
    FactorSpec(
        name="momentum_3m",
        compute=lambda c: momentum_return(c, 63),
        is_alpha=True,
    ),
    FactorSpec(
        name="momentum_12m_1m",
        compute=lambda c: momentum_skip1m(c, 252, 21),
        is_alpha=True,
    ),
    FactorSpec(
        name="vol_compression",
        compute=lambda c: vol_compression_ratio(c, 5, 21),
        is_alpha=True,
    ),
    FactorSpec(
        name="short_term_reversal_5d",
        compute=lambda c: short_term_reversal(c, 5),
        is_alpha=True,
    ),
    FactorSpec(
        name="trend_quality_63d",
        compute=lambda c: trend_quality(c, 63),
        is_alpha=True,
    ),
    FactorSpec(
        name="distance_to_52w_high",
        compute=lambda c: distance_to_52w_high(c, 252),
        is_alpha=True,
    ),
    # Diversifying factors (WS4): structurally anti-correlated with momentum so
    # the blend is not an 80%-momentum book.
    FactorSpec(
        name="reversal_21d",
        compute=lambda c: short_term_reversal(c, 21),
        is_alpha=True,
    ),
    FactorSpec(
        name="low_volatility_63d",
        compute=lambda c: low_volatility(c, 63),
        is_alpha=True,
    ),
    FactorSpec(
        name="mean_reversion_63d",
        compute=lambda c: mean_reversion(c, 63),
        is_alpha=True,
    ),
    FactorSpec(
        name="realized_vol_21d",
        compute=lambda c: realized_vol(c, 21, annualize=True),
        is_alpha=False,
    ),
]
