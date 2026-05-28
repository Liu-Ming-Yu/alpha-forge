"""Core simulated execution fill-model primitives."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable


@dataclass(frozen=True)
class SimulatedLiquidityProfile:
    """Execution liquidity inputs for one simulated order."""

    adv_shares_20d: float
    spread_bps: float


@dataclass(frozen=True)
class SimulatedFillPlan:
    """Planned fill result produced by a simulated execution model."""

    requested_quantity: int
    filled_quantity: int
    fill_price: Decimal
    commission: Decimal
    is_complete: bool
    adv_shares_20d: float
    participation_pct: float
    spread_bps: float
    slippage_bps: float
    implementation_shortfall_bps: float


class ParticipationFillModel:
    """Participation-aware simulated fill planner."""

    def __init__(
        self,
        *,
        liquidity_lookup: Callable[[uuid.UUID, Decimal], SimulatedLiquidityProfile | None],
        max_participation_pct: float,
        fill_price_adjuster: Callable[
            [OrderIntent, Decimal, int, SimulatedLiquidityProfile], Decimal
        ]
        | None = None,
        commission_calculator: Callable[[OrderIntent, Decimal, int], Decimal] | None = None,
        allow_missing_liquidity: bool = True,
        fallback_adv_shares: float = 10_000.0,
        fallback_spread_bps: float = 10.0,
        close_auction_spread_multiplier: float = 1.5,
        stale_price_bps: float = 0.0,
    ) -> None:
        if max_participation_pct <= 0:
            raise ValueError("max_participation_pct must be positive")
        if close_auction_spread_multiplier <= 0:
            raise ValueError("close_auction_spread_multiplier must be positive")
        if stale_price_bps < 0:
            raise ValueError("stale_price_bps must be >= 0")
        self._liquidity_lookup = liquidity_lookup
        self._max_participation_pct = max_participation_pct
        self._fill_price_adjuster = fill_price_adjuster
        self._commission_calculator = commission_calculator
        self._allow_missing_liquidity = allow_missing_liquidity
        self._fallback_adv_shares = fallback_adv_shares
        self._fallback_spread_bps = fallback_spread_bps
        self._close_auction_spread_multiplier = close_auction_spread_multiplier
        self._stale_price_bps = stale_price_bps

    def plan(self, order: OrderIntent, reference_price: Decimal) -> SimulatedFillPlan:
        """Return the fill plan for ``order`` at ``reference_price``."""
        profile = self._liquidity_lookup(order.instrument_id, reference_price)
        if profile is None or profile.adv_shares_20d <= 0:
            if not self._allow_missing_liquidity:
                return self._empty_plan(order, reference_price)
            profile = SimulatedLiquidityProfile(
                adv_shares_20d=max(
                    self._fallback_adv_shares,
                    order.quantity / self._max_participation_pct,
                ),
                spread_bps=self._fallback_spread_bps,
            )
        profile = self._profile_for_order(order, profile)
        fill_reference = self._adverse_reference(order, reference_price)

        max_fill = int(profile.adv_shares_20d * self._max_participation_pct)
        filled_quantity = min(order.quantity, max(max_fill, 0))
        if filled_quantity <= 0:
            return self._empty_plan(order, reference_price, profile=profile)

        fill_price = fill_reference
        if self._fill_price_adjuster is not None:
            fill_price = self._fill_price_adjuster(
                order,
                fill_reference,
                filled_quantity,
                profile,
            )
        fill_price = max(fill_price, Decimal("0.01"))

        commission = Decimal("1.00")
        if self._commission_calculator is not None:
            commission = self._commission_calculator(order, fill_price, filled_quantity)

        participation_pct = (
            filled_quantity / profile.adv_shares_20d if profile.adv_shares_20d > 0 else 0.0
        )
        shortfall_bps = self._implementation_shortfall_bps(
            side=order.side,
            reference_price=reference_price,
            fill_price=fill_price,
        )
        return SimulatedFillPlan(
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            fill_price=fill_price,
            commission=commission,
            is_complete=filled_quantity >= order.quantity,
            adv_shares_20d=profile.adv_shares_20d,
            participation_pct=participation_pct,
            spread_bps=profile.spread_bps,
            slippage_bps=shortfall_bps,
            implementation_shortfall_bps=shortfall_bps,
        )

    def _empty_plan(
        self,
        order: OrderIntent,
        reference_price: Decimal,
        *,
        profile: SimulatedLiquidityProfile | None = None,
    ) -> SimulatedFillPlan:
        adv_shares = 0.0 if profile is None else profile.adv_shares_20d
        spread_bps = self._fallback_spread_bps if profile is None else profile.spread_bps
        return SimulatedFillPlan(
            requested_quantity=order.quantity,
            filled_quantity=0,
            fill_price=reference_price,
            commission=Decimal("0"),
            is_complete=False,
            adv_shares_20d=adv_shares,
            participation_pct=0.0,
            spread_bps=spread_bps,
            slippage_bps=0.0,
            implementation_shortfall_bps=0.0,
        )

    def _profile_for_order(
        self,
        order: OrderIntent,
        profile: SimulatedLiquidityProfile,
    ) -> SimulatedLiquidityProfile:
        if order.order_type not in (OrderType.MOC, OrderType.LOC):
            return profile
        return SimulatedLiquidityProfile(
            adv_shares_20d=profile.adv_shares_20d,
            spread_bps=profile.spread_bps * self._close_auction_spread_multiplier,
        )

    def _adverse_reference(self, order: OrderIntent, reference_price: Decimal) -> Decimal:
        if self._stale_price_bps <= 0:
            return reference_price
        bump = Decimal(str(self._stale_price_bps / 10_000))
        if order.side == OrderSide.BUY:
            return reference_price * (Decimal("1") + bump)
        return max(reference_price * (Decimal("1") - bump), Decimal("0.01"))

    @staticmethod
    def _implementation_shortfall_bps(
        *,
        side: OrderSide,
        reference_price: Decimal,
        fill_price: Decimal,
    ) -> float:
        if reference_price <= 0:
            return 0.0
        if side == OrderSide.BUY:
            return float((fill_price - reference_price) / reference_price * Decimal("10000"))
        return float((reference_price - fill_price) / reference_price * Decimal("10000"))


__all__ = [
    "ParticipationFillModel",
    "SimulatedFillPlan",
    "SimulatedLiquidityProfile",
]
