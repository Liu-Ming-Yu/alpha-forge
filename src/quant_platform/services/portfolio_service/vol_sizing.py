"""Volatility-targeted portfolio constructor.

``VolTargetedPortfolioConstructor`` extends ``LongOnlyPortfolioConstructor``
with risk-parity-inspired position sizing: each instrument's weight is scaled
in inverse proportion to its forecast volatility.

Algorithm
---------
1. Call the base ``LongOnlyPortfolioConstructor.build_targets()`` to get the
   equal-weight base allocation (which respects regime scale, sector caps, etc.).
2. For each instrument in the base weights, look up its forecast annualised vol.
3. Compute an unnormalised risk-parity weight:
       raw_weight_i = base_weight_i × (vol_target / forecast_vol_i)
4. Re-scale raw weights so their sum equals the base total gross exposure
   (preserving the regime-adjusted and risk-limit-aware exposure from step 1).
5. Clip each weight to ``max_single_name_weight``.
6. Recompute cash_target_weight.

Instruments with no vol forecast fall back to their base equal-weight.

Effect: high-vol instruments get smaller weights; low-vol instruments get
proportionally larger weights.  The total gross exposure stays the same as
the equal-weight base; only the distribution across instruments changes.

Usage::

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15)
    bundle = build_feature_bundle(bar_data)
    constructor.set_vol_forecasts(bundle.vol_forecasts)
    target = constructor.build_targets(signals, regime, account, limits)
"""

from __future__ import annotations

import uuid
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore

log = structlog.get_logger(__name__)

_QUANTIZE = Decimal("0.0001")  # 4 decimal places for weight precision


class VolTargetedPortfolioConstructor(LongOnlyPortfolioConstructor):
    """Equal-weight baseline with per-instrument volatility-parity scaling.

    Inherits all regime scaling, sector caps, and single-name weight limits
    from ``LongOnlyPortfolioConstructor``.  After the base weights are built,
    applies inverse-volatility scaling to redistribute within the same
    gross exposure envelope.

    Args:
        vol_target: Target annualised volatility per position, as a decimal
            fraction (default 0.15 = 15%).  Instruments with forecast vol
            equal to ``vol_target`` receive unchanged weight.  Lower-vol
            instruments receive more weight; higher-vol receive less.
        min_vol_floor: Minimum vol used in the denominator to prevent
            over-sizing very low-vol instruments.  Default 0.05 (5%).
        **kwargs: Passed to ``LongOnlyPortfolioConstructor``.

    Must never:
        Use vol forecasts from a prior cycle (stale forecasts are worse than
        equal-weight).  Always call ``set_vol_forecasts()`` before each cycle.
        Violate any ``RiskLimits`` from the base constructor.
    """

    def __init__(
        self,
        vol_target: float = 0.15,
        min_vol_floor: float = 0.05,
        top_n: int = 10,
        min_score_threshold: float = 0.0,
        sector_map: dict[uuid.UUID, str] | None = None,
        regime_scales: dict[RegimeLabel, Decimal] | None = None,
    ) -> None:
        super().__init__(
            top_n=top_n,
            min_score_threshold=min_score_threshold,
            sector_map=sector_map,
            regime_scales=regime_scales,
        )
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")
        if min_vol_floor <= 0:
            raise ValueError(f"min_vol_floor must be positive, got {min_vol_floor}")
        self._vol_target = Decimal(str(vol_target))
        self._min_vol_floor = Decimal(str(min_vol_floor))
        self._vol_forecasts: dict[uuid.UUID, Decimal] = {}

    def set_vol_forecasts(self, forecasts: dict[uuid.UUID, float]) -> None:
        """Update per-instrument vol forecasts for the next cycle.

        Call this once per rebalance with forecasts from ``FeatureBundle.vol_forecasts``
        before calling ``build_targets()``.  Instruments not present in ``forecasts``
        fall back to equal-weight from the base constructor.

        Args:
            forecasts: Instrument_id → annualised realised vol (decimal fraction).
                Negative or zero values are silently dropped.
        """
        self._vol_forecasts = {k: Decimal(str(v)) for k, v in forecasts.items() if v > 0}
        log.debug(
            "vol_constructor.forecasts_updated",
            n_instruments=len(self._vol_forecasts),
        )

    def build_targets(
        self,
        signals: list[SignalScore],
        regime: RegimeState,
        account: AccountSnapshot,
        limits: RiskLimits,
    ) -> PortfolioTarget:
        """Build vol-scaled PortfolioTarget.

        Delegates to the base constructor for filtering, ranking, regime scaling,
        and sector enforcement.  Then applies inverse-volatility scaling.

        Returns the base target unchanged when:
        - No vol forecasts have been loaded (call ``set_vol_forecasts()`` first).
        - The base target has no investable positions (e.g., CRISIS regime or no signals).
        """
        base = super().build_targets(signals, regime, account, limits)

        if not self._vol_forecasts or not base.weights:
            return base

        # Compute base total gross exposure (≤ max_gross_exposure × regime_scale)
        base_total = sum(base.weights.values(), Decimal("0"))
        if base_total <= Decimal("0"):
            return base

        # Step 1: apply vol scaling to each base weight
        raw_scaled: dict[uuid.UUID, Decimal] = {}
        for instr_id, base_w in base.weights.items():
            forecast_vol = self._vol_forecasts.get(instr_id)
            if forecast_vol is not None and forecast_vol > Decimal("0"):
                # Clamp forecast vol to min_vol_floor to avoid over-sizing
                effective_vol = max(forecast_vol, self._min_vol_floor)
                scale = self._vol_target / effective_vol
                raw_scaled[instr_id] = base_w * scale
            else:
                # No forecast: use base weight unchanged
                raw_scaled[instr_id] = base_w

        # Step 2: renormalize so that the scaled weights sum to base_total
        # This preserves the regime-adjusted gross exposure from the base constructor.
        scaled_total = sum(raw_scaled.values(), Decimal("0"))
        if scaled_total <= Decimal("0"):
            return base

        norm_factor = base_total / scaled_total
        normalized: dict[uuid.UUID, Decimal] = {}
        for instr_id, w in raw_scaled.items():
            max_single_name_weight = Decimal(str(limits.max_single_name_weight))
            capped = min(w * norm_factor, max_single_name_weight)
            normalized[instr_id] = capped.quantize(_QUANTIZE, rounding=ROUND_HALF_UP)

        # Step 3: final cash target
        invested = sum(normalized.values())
        cash_target = max(Decimal("0"), Decimal("1") - invested).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )

        notes = list(base.construction_notes)
        notes.append(
            f"vol_targeted(vol_target={self._vol_target}, "
            f"floor={self._min_vol_floor}, "
            f"n_forecasts={len(self._vol_forecasts)})"
        )

        log.info(
            "vol_constructor.built",
            n_names=len(normalized),
            base_total=str(base_total),
            scaled_total=str(invested),
            vol_target=str(self._vol_target),
        )

        return PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=base.strategy_run_id,
            as_of=base.as_of,
            regime_id=base.regime_id,
            weights=normalized,
            cash_target_weight=cash_target,
            construction_notes=notes,
        )
