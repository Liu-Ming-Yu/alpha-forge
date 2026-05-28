"""Default V2 execution router."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.orders import (
    CancelReplaceRequest,
    ExecutionTactic,
    OrderIntent,
    OrderType,
    VenueRoute,
)
from quant_platform.core.domain.production import ExecutionTacticPolicy

if TYPE_CHECKING:
    from quant_platform.core.contracts import BrokerOrderRoutingGateway


class DefaultExecutionRouter:
    """Map approved order intents to V2 EMS tactics."""

    def __init__(
        self,
        policy: ExecutionTacticPolicy | None = None,
        *,
        broker: BrokerOrderRoutingGateway | None = None,
        venue: str = "IBKR_SMART",
        size_urgency_thresholds: list[tuple[int, float]] | None = None,
    ) -> None:
        self._policy = policy or ExecutionTacticPolicy()
        self._broker = broker
        self._venue = venue
        # Sorted ascending by max_shares so the first match gives the correct tier.
        self._size_urgency_thresholds: list[tuple[int, float]] = sorted(
            size_urgency_thresholds or [(500, 0.25), (2000, 0.50), (10000, 0.75), (50000, 1.0)],
            key=lambda t: t[0],
        )

    def route(self, intent: OrderIntent) -> VenueRoute:
        if intent.order_type == OrderType.MOC:
            tactic = ExecutionTactic.CLOSE_AUCTION_MOC
            urgency = Decimal("1")
        elif intent.order_type == OrderType.LOC:
            tactic = ExecutionTactic.CLOSE_AUCTION_LOC
            urgency = Decimal("0.75")
        elif self._policy.passive_limit_enabled and intent.order_type == OrderType.LIMIT:
            tactic = ExecutionTactic.PASSIVE_LIMIT
            urgency = Decimal("0.25")
        else:
            tactic = ExecutionTactic.URGENCY_LIMIT
            urgency = Decimal(
                str(_size_urgency(abs(intent.quantity), self._size_urgency_thresholds))
            )
        return VenueRoute(
            route_id=uuid.uuid4(),
            venue=self._venue,
            tactic=tactic,
            max_participation_rate=Decimal(str(self._policy.max_adv_participation_pct)),
            urgency=urgency,
        )

    async def cancel_replace(self, request: CancelReplaceRequest) -> None:
        """Cancel old order; replacement intent creation is owned by the caller."""
        if self._broker is None:
            raise RuntimeError("cancel_replace requires a broker routing gateway")
        await self._broker.cancel_order(request.broker_order_id)


def _size_urgency(quantity: int, thresholds: list[tuple[int, float]]) -> float:
    """Return urgency score for an order of `quantity` shares.

    Thresholds must be sorted ascending by max_shares.  The first tier whose
    max_shares >= quantity applies; orders exceeding all tiers get urgency 1.0.
    """
    for max_shares, urgency in thresholds:
        if quantity <= max_shares:
            return urgency
    return 1.0
