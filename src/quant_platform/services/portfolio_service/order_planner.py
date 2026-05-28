"""Portfolio target → order intent conversion.

PortfolioTargetOrderPlanner converts a PortfolioTarget (expressed as target
weights) into executable OrderIntents.  Sells are sequenced before buys so
that the cash freed by sales is available for new purchases, satisfying the
cash-account rule that settled cash must exist before a buy is approved.

Rebalance threshold:
    Small delta weights (|target_weight − current_weight| < threshold) are
    skipped to avoid excessive turnover on noise.  The threshold is expressed
    as a fraction of NAV (default 1%).

Share rounding:
    Dollar deltas are divided by the instrument's market price and truncated
    to whole shares (int()).  Resulting allocations of 0 shares are silently
    dropped.

Price resolution:
    1. market_prices dict passed to plan() — preferred source for current data.
    2. PositionSnapshot.market_price — fallback for positions already held.
    3. No price → instrument skipped with a warning.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import Clock
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

log = structlog.get_logger(__name__)


class PortfolioTargetOrderPlanner:
    """Convert PortfolioTarget weights into sell-before-buy OrderIntents.

    Args:
        clock: Injectable time source; used for OrderIntent.created_at.
        rebalance_threshold: Minimum |delta_weight| (as a fraction of NAV)
            required to generate an order.  Default 0.01 (1% of NAV).
        order_type: Order type for all generated intents.
            Use MARKET for live sessions; LIMIT for simulated sessions where
            a precise fill price is needed.  When LIMIT is selected, the
            price from market_prices (or the position snapshot) becomes the
            limit_price on the OrderIntent.

    Must never:
        Access the broker or the cash ledger directly.
        Perform cash sufficiency checks (that is the PreTradeGate's job).
        Generate orders for instruments not in target.weights and not currently
        held (instruments at zero current weight with zero target weight).
    """

    def __init__(
        self,
        clock: Clock,
        rebalance_threshold: Decimal = Decimal("0.01"),
        order_type: OrderType = OrderType.LIMIT,
    ) -> None:
        self._clock = clock
        self._rebalance_threshold = rebalance_threshold
        self._order_type = order_type

    def plan(
        self,
        target: PortfolioTarget,
        account: AccountSnapshot,
        market_prices: dict[uuid.UUID, Decimal],
        strategy_run_id: uuid.UUID,
    ) -> list[OrderIntent]:
        """Convert target weights to sell-before-buy OrderIntents.

        Args:
            target: The PortfolioTarget produced by PortfolioConstructor.
            account: Current account state.  Positions in the snapshot supply
                current weights; instruments not in positions have weight 0.
            market_prices: Current market prices keyed by instrument_id.
                Required for accurate share-count calculation.  Falls back
                to PositionSnapshot.market_price for held positions if an
                instrument is absent from this dict.
            strategy_run_id: Written to every generated OrderIntent for audit
                attribution.

        Returns:
            A list of OrderIntents with all sells preceding all buys.
            Instruments whose |delta_weight| falls below the rebalance
            threshold are omitted.  Empty list when NAV is zero or when all
            deltas are below threshold.
        """
        nav = account.net_asset_value
        if nav <= Decimal("0"):
            return []

        now = self._clock.now()

        # Current weight for each held position.
        current_weights: dict[uuid.UUID, Decimal] = {
            pos.instrument_id: pos.market_value / nav for pos in account.positions
        }

        def _price(instr_id: uuid.UUID) -> Decimal | None:
            """Resolve market price: market_prices dict first, position fallback."""
            p = market_prices.get(instr_id)
            if p is not None:
                return p
            for pos in account.positions:
                if pos.instrument_id == instr_id:
                    return pos.market_price
            return None

        # Consider every instrument that is either targeted or currently held.
        all_ids: set[uuid.UUID] = set(target.weights.keys()) | set(current_weights.keys())

        sells: list[OrderIntent] = []
        buys: list[OrderIntent] = []

        for instr_id in all_ids:
            target_w = target.weights.get(instr_id, Decimal("0"))
            current_w = current_weights.get(instr_id, Decimal("0"))
            delta_w = target_w - current_w

            # Skip sub-threshold rebalances to limit turnover.
            if abs(delta_w) < self._rebalance_threshold:
                continue

            price = _price(instr_id)
            if price is None or price <= Decimal("0"):
                log.warning(
                    "order_planner.no_price",
                    instrument_id=str(instr_id),
                    delta_weight=str(delta_w),
                )
                continue

            dollar_delta = abs(delta_w) * nav
            shares = int(dollar_delta / price)
            if shares < 1:
                continue

            side = OrderSide.SELL if delta_w < Decimal("0") else OrderSide.BUY
            limit_price = price if self._order_type == OrderType.LIMIT else None

            intent = OrderIntent(
                order_id=uuid.uuid4(),
                strategy_run_id=strategy_run_id,
                portfolio_target_id=target.target_id,
                instrument_id=instr_id,
                side=side,
                quantity=shares,
                order_type=self._order_type,
                time_in_force=TimeInForce.DAY,
                created_at=now,
                limit_price=limit_price,
            )

            if side == OrderSide.SELL:
                sells.append(intent)
            else:
                buys.append(intent)

        log.info(
            "order_planner.summary",
            sells=len(sells),
            buys=len(buys),
            target_id=str(target.target_id),
        )

        # Sells first — free cash before committing it.
        return sells + buys
