"""Slippage and commission models for realistic backtest simulation.

Production backtests must account for execution costs to prevent
over-optimistic alpha estimates.  These models are used by the
backtest engine to adjust fills from SimulatedBrokerGateway.

SlippageModel protocol:
    estimate_slippage(order_shares, adv_shares, spread_bps, side) -> bps

Concrete implementations:
    FixedSlippageModel       - constant basis-point spread
    SquareRootSlippageModel  - participation-rate-based market impact
    IBKRCommissionSchedule   - IBKR tiered commission for US equities
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


class SlippageSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SlippageModel(Protocol):
    """Estimate execution slippage in basis points."""

    def estimate_slippage(
        self,
        order_shares: int,
        adv_shares: float,
        spread_bps: float,
        side: SlippageSide,
    ) -> float:
        """Return estimated one-way slippage in basis points.

        Args:
            order_shares: Number of shares in the order.
            adv_shares: 20-day average daily volume in shares.
            spread_bps: Estimated bid-ask spread in basis points.
            side: Buy or sell (buys get adverse slippage, sells get adverse).
        """
        ...


@dataclass(frozen=True)
class FixedSlippageModel:
    """Constant slippage regardless of order size or market conditions.

    Useful as a conservative floor estimate or for quick research iteration.
    Default: 5 bps (one-way).
    """

    fixed_bps: float = 5.0

    def estimate_slippage(
        self,
        order_shares: int,
        adv_shares: float,
        spread_bps: float,
        side: SlippageSide,
    ) -> float:
        return self.fixed_bps


@dataclass(frozen=True)
class SquareRootSlippageModel:
    """Market impact model based on the square-root participation rate.

    The classic Almgren-Chriss-style approximation:
        impact_bps = sigma_daily_bps * eta * sqrt(order_shares / adv_shares) + spread_bps / 2

    Args:
        sigma_daily_bps: Typical daily volatility in bps (default 100 = 1%).
        eta: Impact coefficient (default 0.6, calibrated to US large-cap).
        min_slippage_bps: Floor slippage to prevent unrealistic zero-cost fills.
    """

    sigma_daily_bps: float = 100.0
    eta: float = 0.6
    min_slippage_bps: float = 1.0

    def estimate_slippage(
        self,
        order_shares: int,
        adv_shares: float,
        spread_bps: float,
        side: SlippageSide,
    ) -> float:
        if adv_shares <= 0 or order_shares <= 0:
            return self.min_slippage_bps

        participation = order_shares / adv_shares
        impact = self.sigma_daily_bps * self.eta * math.sqrt(participation)
        half_spread = spread_bps / 2.0
        total = impact + half_spread

        return max(total, self.min_slippage_bps)


@dataclass(frozen=True)
class IBKRCommissionSchedule:
    """IBKR tiered commission schedule for US equities.

    IBKR charges per-share with a per-order minimum and maximum.
    Default values reflect the standard tiered pricing as of 2024.

    Args:
        per_share: Commission per share in USD.
        min_per_order: Minimum commission per order in USD.
        max_pct_of_value: Maximum commission as a fraction of trade value.
    """

    per_share: Decimal = Decimal("0.005")
    min_per_order: Decimal = Decimal("1.00")
    max_pct_of_value: Decimal = Decimal("0.01")

    def compute(self, shares: int, price: Decimal) -> Decimal:
        """Return the commission in USD for a given fill."""
        raw = self.per_share * Decimal(str(shares))
        trade_value = Decimal(str(shares)) * price
        cap = trade_value * self.max_pct_of_value

        commission = max(raw, self.min_per_order)
        commission = min(commission, cap) if cap > 0 else commission
        return commission
