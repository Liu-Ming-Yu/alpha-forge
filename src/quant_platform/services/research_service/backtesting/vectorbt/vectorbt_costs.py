"""Execution-cost helpers for the VectorBT backtest facade."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from ..simple.backtest_execution_model import (
    spread_bps_for_price,
)
from ..slippage import SlippageSide

if TYPE_CHECKING:
    import uuid

    from ..simple.backtest_execution_model import (
        BacktestExecutionModel,
    )
    from ..slippage import (
        IBKRCommissionSchedule,
        SlippageModel,
    )


class VectorBTCostMixin:
    """Compatibility cost helpers backed by the vectorized execution model."""

    _slippage_model: SlippageModel
    _commission_schedule: IBKRCommissionSchedule
    _execution_model: BacktestExecutionModel

    def _compute_slippage_frac(self, instrument_id: uuid.UUID) -> float:
        """Return a representative slippage fraction for this instrument."""
        adv_shares, spread_bps = self._lookup_liquidity_params(instrument_id, Decimal("100"))
        slippage_bps = self._slippage_model.estimate_slippage(
            order_shares=max(1, int(adv_shares * 0.001)),
            adv_shares=adv_shares,
            spread_bps=spread_bps,
            side=SlippageSide.BUY,
        )
        return float(slippage_bps / 10_000.0)

    def _compute_commission_frac(self) -> float:
        """Return commission as a fraction of trade notional."""
        shares = 100
        price = Decimal("100")
        commission_usd = float(self._commission_schedule.compute(shares, price))
        notional = shares * float(price)
        return commission_usd / notional if notional > 0 else 0.0

    def _lookup_liquidity_params(
        self,
        instrument_id: uuid.UUID,
        reference_price: Decimal,
    ) -> tuple[float, float]:
        """Return (adv_shares, spread_bps) for slippage computation."""
        return self._execution_model.lookup_liquidity_params(instrument_id, reference_price)

    @staticmethod
    def _spread_bps_for_price(reference_price: Decimal) -> float:
        """Heuristic price-bucket spread model."""
        return spread_bps_for_price(reference_price)


__all__ = ["VectorBTCostMixin"]
