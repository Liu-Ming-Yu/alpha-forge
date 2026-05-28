"""Runtime wiring for passive limit reprice coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import structlog

from quant_platform.services.execution_service.orders.router import DefaultExecutionRouter
from quant_platform.services.execution_service.passive_reprice import (
    PassiveLimitRepriceCoordinator,
)

if TYPE_CHECKING:
    import uuid
    from decimal import Decimal

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.contracts import OrderStateStore
    from quant_platform.core.domain.orders import OrderIntent
    from quant_platform.services.execution_service.passive_reprice.passive_reprice_models import (
        PassiveRepriceDecision,
    )

log = structlog.get_logger(__name__)


async def run_passive_reprice_once(
    *,
    session: Session,
    market_prices: dict[uuid.UUID, Decimal],
) -> list[PassiveRepriceDecision]:
    """Run one conservative passive-reprice pass for the session.

    Runtime wiring is cancel-only.  Replacement intents remain outside this
    path until they can re-enter approval, cash reservation, and attribution.
    """
    policy = session.execution_tactic_policy
    if not policy.passive_limit_enabled:
        return []

    capabilities = getattr(session.trading_broker, "capabilities", None)
    if capabilities is not None and not capabilities.supports_order_cancellation:
        log.warning(
            "passive_reprice.unsupported_broker",
            provider=capabilities.provider,
        )
        return []

    coordinator = PassiveLimitRepriceCoordinator(
        policy=policy,
        router=DefaultExecutionRouter(policy=policy, broker=session.trading_broker),
        broker=session.broker,
        order_repo=session.order_repo,
        order_state=cast("OrderStateStore | None", session.v2_order_state),
        clock=session.clock,
        reference_price_lookup=lambda intent: _reference_price(intent, market_prices),
    )
    return await coordinator.run_once()


def _reference_price(
    intent: OrderIntent,
    market_prices: dict[uuid.UUID, Decimal],
) -> Decimal | None:
    price = market_prices.get(intent.instrument_id)
    if price is None or price <= 0:
        return None
    return price
