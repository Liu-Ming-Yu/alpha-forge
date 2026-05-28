"""Typed campaign portfolio data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid


@dataclass(frozen=True)
class CampaignPortfolioConfig:
    """Effective risk controls for campaign eligibility portfolio returns."""

    mode: str = "runtime-long-only"
    top_n: int = 10
    vol_target: float = 0.15
    vol_floor: float = 0.05
    vol_lookback_days: int = 63
    max_gross_exposure: float = 0.60
    min_cash_buffer: float = 0.05
    max_single_name_weight: float = 0.05
    max_daily_turnover: float = 0.20
    max_position_change: float = 0.05
    no_trade_band: float = 0.0
    """Hold the current weight when the desired change is below this band.

    Cost-aware hysteresis: suppresses churn from small score wiggles, the
    dominant driver of the turnover that erodes slippage-adjusted Sharpe.
    ``0.0`` disables the band (rebalance every name every day).
    """
    rebalance_interval_days: int = 1
    """Rebalance only every Nth observation day; carry weights forward between.

    ``1`` rebalances daily (original behaviour); larger values cut turnover.
    """

    def __post_init__(self) -> None:
        if self.mode != "runtime-long-only":
            raise ValueError(f"unsupported campaign portfolio mode: {self.mode}")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        for field_name in (
            "vol_target",
            "vol_floor",
            "max_gross_exposure",
            "max_single_name_weight",
            "max_daily_turnover",
            "max_position_change",
        ):
            if float(getattr(self, field_name)) <= 0.0:
                raise ValueError(f"{field_name} must be > 0")
        if not 0.0 <= self.min_cash_buffer < 1.0:
            raise ValueError("min_cash_buffer must be in [0, 1)")
        if self.vol_lookback_days < 1:
            raise ValueError("vol_lookback_days must be >= 1")
        if self.no_trade_band < 0.0:
            raise ValueError("no_trade_band must be >= 0")
        if self.rebalance_interval_days < 1:
            raise ValueError("rebalance_interval_days must be >= 1")

    @property
    def effective_max_gross_cap(self) -> float:
        """Maximum gross reachable before missing-positive-name cash drag."""
        return min(
            float(self.max_gross_exposure),
            1.0 - float(self.min_cash_buffer),
            int(self.top_n) * float(self.max_single_name_weight),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "top_n": int(self.top_n),
            "vol_target": float(self.vol_target),
            "vol_floor": float(self.vol_floor),
            "vol_lookback_days": int(self.vol_lookback_days),
            "max_gross_exposure": float(self.max_gross_exposure),
            "min_cash_buffer": float(self.min_cash_buffer),
            "max_single_name_weight": float(self.max_single_name_weight),
            "max_daily_turnover": float(self.max_daily_turnover),
            "max_position_change": float(self.max_position_change),
            "no_trade_band": float(self.no_trade_band),
            "rebalance_interval_days": int(self.rebalance_interval_days),
            "effective_max_gross_cap": float(self.effective_max_gross_cap),
            "effective_gross_cap_reason": (
                "min(max_gross_exposure, 1 - min_cash_buffer, top_n * max_single_name_weight)"
            ),
            "fold_boundary_position_policy": "carry_positions_across_folds",
        }


@dataclass(frozen=True)
class FoldVolatilityScale:
    """Fold-level exposure scale fit only from training-window returns."""

    exposure_scale: float
    train_realized_vol: float
    train_effective_vol: float
    raw_vol_scale: float
    train_observations: int

    def to_payload(self) -> dict[str, float]:
        return {
            "exposure_scale": float(self.exposure_scale),
            "train_realized_vol": float(self.train_realized_vol),
            "train_effective_vol": float(self.train_effective_vol),
            "raw_vol_scale": float(self.raw_vol_scale),
            "train_observations": float(self.train_observations),
        }


@dataclass(frozen=True)
class PortfolioEvaluation:
    """Daily portfolio return series plus state and diagnostics."""

    daily_returns: tuple[float, ...]
    daily_ics: tuple[tuple[str, float], ...]
    daily_turnover: tuple[float, ...]
    final_weights: dict[uuid.UUID, float]
    day_diagnostics: tuple[dict[str, object], ...]


__all__ = [
    "CampaignPortfolioConfig",
    "FoldVolatilityScale",
    "PortfolioEvaluation",
]
