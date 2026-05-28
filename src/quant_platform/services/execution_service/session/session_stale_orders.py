"""Stale open-order cleanup for broker session supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import BrokerOrder, TimeInForce

if TYPE_CHECKING:
    import uuid
    from datetime import datetime, timedelta

    from quant_platform.core.contracts import (
        BrokerOrderRoutingGateway,
        BrokerSessionGateway,
        OrderRepository,
    )

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StaleOrderCleanupResult:
    cancelled: int
    broker_errors: int


async def cleanup_stale_orders(
    *,
    session_gateway: BrokerSessionGateway,
    order_gateway: BrokerOrderRoutingGateway,
    order_repo: OrderRepository,
    now: datetime,
    strategy_run_id: uuid.UUID,
    day_threshold: timedelta,
    gtc_threshold: timedelta | None = None,
) -> StaleOrderCleanupResult:
    """Cancel stale DAY orders and optionally stale GTC orders."""
    if not order_gateway.capabilities.supports_order_cancellation:
        return StaleOrderCleanupResult(cancelled=0, broker_errors=0)

    open_orders = await session_gateway.fetch_open_orders()
    cancelled = 0
    broker_errors = 0
    day_result = await _cancel_by_tif(
        open_orders=open_orders,
        order_gateway=order_gateway,
        order_repo=order_repo,
        now=now,
        threshold=day_threshold,
        tif=TimeInForce.DAY,
        log_event="session_supervisor.cancel_stale_failed",
    )
    cancelled += day_result.cancelled
    broker_errors += day_result.broker_errors
    if gtc_threshold is not None:
        gtc_result = await _cancel_by_tif(
            open_orders=open_orders,
            order_gateway=order_gateway,
            order_repo=order_repo,
            now=now,
            threshold=gtc_threshold,
            tif=TimeInForce.GTC,
            log_event="session_supervisor.cancel_stale_gtc_failed",
        )
        cancelled += gtc_result.cancelled
        broker_errors += gtc_result.broker_errors

    if cancelled:
        log.info(
            "session_supervisor.stale_orders_cancelled",
            count=cancelled,
            strategy_run_id=str(strategy_run_id),
        )
    return StaleOrderCleanupResult(cancelled=cancelled, broker_errors=broker_errors)


async def _cancel_by_tif(
    *,
    open_orders: list[BrokerOrder],
    order_gateway: BrokerOrderRoutingGateway,
    order_repo: OrderRepository,
    now: datetime,
    threshold: timedelta,
    tif: TimeInForce,
    log_event: str,
) -> StaleOrderCleanupResult:
    cancelled = 0
    broker_errors = 0
    for broker_order in open_orders:
        if broker_order.broker_order_id is None:
            continue
        if now - broker_order.last_updated_at < threshold:
            continue
        intent = await order_repo.get_intent(broker_order.order_id)
        if intent is None or intent.time_in_force != tif:
            continue
        try:
            await order_gateway.cancel_order(broker_order.broker_order_id)
            cancelled += 1
        except Exception as exc:
            broker_errors += 1
            log.warning(
                log_event,
                order_id=str(broker_order.order_id),
                broker_order_id=broker_order.broker_order_id,
                error=str(exc),
            )
    return StaleOrderCleanupResult(cancelled=cancelled, broker_errors=broker_errors)
