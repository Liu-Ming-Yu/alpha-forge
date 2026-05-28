"""Simulated execution-cost wiring for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.algorithms.simulated_execution import (
    ParticipationFillModel,
    SimulatedLiquidityProfile,
)

from ..slippage import (
    IBKRCommissionSchedule,
    SlippageModel,
    SlippageSide,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import BacktestReplayBroker, LiquidityProfileProvider
    from quant_platform.core.contracts.data import LiquidityProfileSnapshot
    from quant_platform.core.domain.orders import OrderIntent

log = structlog.get_logger(__name__)

FALLBACK_ADV_SHARES = 10_000.0
FALLBACK_SPREAD_BPS = 10.0
CLOSE_AUCTION_SPREAD_MULTIPLIER = 1.5
STALE_PRICE_BPS = 1.0


@dataclass
class BacktestExecutionModel:
    """Configure simulated fills and resolve liquidity/cost assumptions."""

    settings: PlatformSettings
    slippage_model: SlippageModel
    commission_schedule: IBKRCommissionSchedule
    universe_manager: LiquidityProfileProvider | None
    fallback_logged: set[uuid.UUID]

    def configure_simulated_execution_model(self, broker: BacktestReplayBroker) -> None:
        """Apply participation, slippage, and commission models to fills."""

        def _liquidity(
            instrument_id: uuid.UUID,
            reference_price: Decimal,
        ) -> SimulatedLiquidityProfile:
            adv_shares, spread_bps = self.lookup_liquidity_params(
                instrument_id,
                reference_price,
            )
            return SimulatedLiquidityProfile(
                adv_shares_20d=adv_shares,
                spread_bps=spread_bps,
            )

        def _adjust_price(
            intent: OrderIntent,
            reference_price: Decimal,
            filled_quantity: int,
            liquidity: SimulatedLiquidityProfile,
        ) -> Decimal:
            side = SlippageSide.BUY if intent.side.value == "buy" else SlippageSide.SELL
            slippage_bps = self.slippage_model.estimate_slippage(
                order_shares=filled_quantity,
                adv_shares=liquidity.adv_shares_20d,
                spread_bps=liquidity.spread_bps,
                side=side,
            )
            bump = Decimal(str(slippage_bps / 10_000))
            if side == SlippageSide.SELL:
                bump = -bump
            adjusted = reference_price * (Decimal("1") + bump)
            return max(adjusted, Decimal("0.01"))

        def _commission(
            intent: OrderIntent,
            fill_price: Decimal,
            filled_quantity: int,
        ) -> Decimal:
            del intent
            return self.commission_schedule.compute(filled_quantity, fill_price)

        broker.configure_execution_model(
            ParticipationFillModel(
                liquidity_lookup=_liquidity,
                max_participation_pct=self.settings.liquidity.adv_participation_pct,
                allow_missing_liquidity=self.settings.liquidity.allow_missing_profile,
                fallback_adv_shares=FALLBACK_ADV_SHARES,
                fallback_spread_bps=FALLBACK_SPREAD_BPS,
                close_auction_spread_multiplier=CLOSE_AUCTION_SPREAD_MULTIPLIER,
                stale_price_bps=STALE_PRICE_BPS,
                fill_price_adjuster=_adjust_price,
                commission_calculator=_commission,
            )
        )

    def configure_simulated_cost_model(self, broker: BacktestReplayBroker) -> None:
        """Fill cost hook for tests and external callers."""

        def _adjust_price(intent: OrderIntent, reference_price: Decimal) -> Decimal:
            side = SlippageSide.BUY if intent.side.value == "buy" else SlippageSide.SELL
            adv_shares, spread_bps = self.lookup_liquidity_params(
                intent.instrument_id,
                reference_price,
            )
            slippage_bps = self.slippage_model.estimate_slippage(
                order_shares=intent.quantity,
                adv_shares=adv_shares,
                spread_bps=spread_bps,
                side=side,
            )
            bump = Decimal(str(slippage_bps / 10_000))
            if side == SlippageSide.SELL:
                bump = -bump
            adjusted = reference_price * (Decimal("1") + bump)
            return max(adjusted, Decimal("0.01"))

        def _commission(intent: OrderIntent, fill_price: Decimal) -> Decimal:
            return self.commission_schedule.compute(intent.quantity, fill_price)

        broker.configure_execution_cost_model(
            fill_price_adjuster=_adjust_price,
            commission_calculator=_commission,
        )

    def lookup_liquidity_params(
        self,
        instrument_id: uuid.UUID,
        reference_price: Decimal,
    ) -> tuple[float, float]:
        """Return ``(adv_shares, spread_bps)`` for slippage computation."""
        profile: LiquidityProfileSnapshot | None = None
        if self.universe_manager is not None:
            profile = self.universe_manager.get_profile(instrument_id)

        if profile is None:
            if instrument_id not in self.fallback_logged:
                self.fallback_logged.add(instrument_id)
                log.info(
                    "backtest.slippage.fallback",
                    instrument_id=str(instrument_id),
                    adv_shares=FALLBACK_ADV_SHARES,
                    spread_bps=FALLBACK_SPREAD_BPS,
                )
            return FALLBACK_ADV_SHARES, FALLBACK_SPREAD_BPS

        adv_shares = max(profile.adv_shares_20d, 1.0)
        spread_bps = spread_bps_for_price(reference_price)
        return adv_shares, spread_bps


def spread_bps_for_price(reference_price: Decimal) -> float:
    """Heuristic price-bucket spread model."""
    p = float(reference_price)
    if p < 5:
        return 20.0
    if p < 20:
        return 8.0
    if p < 100:
        return 4.0
    return 2.0


def slippage_bps_from_prices(
    *,
    side: str,
    model_price: Decimal,
    fill_price: Decimal,
) -> float:
    """Return execution slippage in basis points."""
    if model_price <= 0:
        return 0.0
    if side == "buy":
        return float((fill_price - model_price) / model_price * Decimal("10000"))
    return float((model_price - fill_price) / model_price * Decimal("10000"))
