"""Pure mapping helpers for account-level V2 execution."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio import PortfolioTarget
from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    OrderAllocation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from quant_platform.core.domain.orders import OrderIntent

UNKNOWN_PARENT_PROPOSAL_ID = uuid.UUID(int=0)


def combined_to_portfolio_target(
    combined: CombinedPortfolioTarget,
    *,
    strategy_run_id: uuid.UUID,
) -> PortfolioTarget:
    """Adapt a merged account target to the portfolio optimizer contract."""
    return PortfolioTarget(
        target_id=combined.target_id,
        strategy_run_id=strategy_run_id,
        as_of=combined.as_of,
        regime_id=uuid.uuid5(uuid.NAMESPACE_URL, str(combined.target_id)),
        weights=combined.weights,
        cash_target_weight=combined.cash_target_weight,
        construction_notes=list(combined.construction_notes),
    )


def parent_proposal_ids_for_orders(
    planned: Sequence[OrderIntent],
    combined: CombinedPortfolioTarget,
) -> tuple[uuid.UUID, ...]:
    """Return the source proposal id for each planned order."""
    proposal_by_instrument: dict[uuid.UUID, uuid.UUID] = {}
    for contribution in combined.contributions:
        for instrument_id in contribution.weights:
            proposal_by_instrument.setdefault(instrument_id, contribution.contribution_id)
    return tuple(
        proposal_by_instrument.get(intent.instrument_id, UNKNOWN_PARENT_PROPOSAL_ID)
        for intent in planned
    )


def order_allocations_for_intents(
    intents: Sequence[OrderIntent],
    target: CombinedPortfolioTarget,
    market_prices: dict[uuid.UUID, Decimal],
    *,
    allocation_id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> list[OrderAllocation]:
    """Build engine-attribution rows for approved merged orders."""
    allocations: list[OrderAllocation] = []
    for intent in intents:
        price = market_prices.get(intent.instrument_id)
        notional = price * Decimal(intent.quantity) if price is not None else None
        for contribution in target.contributions:
            weight = contribution.weights.get(intent.instrument_id, Decimal("0"))
            if weight <= Decimal("0"):
                continue
            allocations.append(
                OrderAllocation(
                    allocation_id=allocation_id_factory(),
                    order_id=intent.order_id,
                    engine_name=contribution.engine_name,
                    strategy_run_id=contribution.strategy_run_id,
                    instrument_id=intent.instrument_id,
                    allocated_weight=weight,
                    allocated_notional=notional,
                )
            )
    return allocations


__all__ = [
    "UNKNOWN_PARENT_PROPOSAL_ID",
    "combined_to_portfolio_target",
    "order_allocations_for_intents",
    "parent_proposal_ids_for_orders",
]
