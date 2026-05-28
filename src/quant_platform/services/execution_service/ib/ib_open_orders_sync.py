"""IB open-order sync and lifecycle-map rebuild helpers."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.core.contracts import BrokerAck
from quant_platform.core.domain.orders.lifecycle import BrokerOrphanDetected

if TYPE_CHECKING:
    from quant_platform.core.domain.orders import BrokerOrder

log = structlog.get_logger(__name__)


async def fetch_open_orders_sync(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    con_id_to_instrument: dict[int, uuid.UUID],
    submitted: dict[uuid.UUID, BrokerAck],
    execution_policy: object,
    orphan_ttl_minutes: int,
) -> list[BrokerOrder]:
    """Fetch open IB orders and rebuild lifecycle maps after reconnect."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    execution_policy_any = cast("Any", execution_policy)
    loop = asyncio.get_running_loop()
    wrapper_any._open_orders = []
    wrapper_any._open_orders_done = loop.create_future()
    # Reset the staging map so this round's openOrder callbacks are the
    # only source of truth for the lifecycle-map rebuild below.
    with wrapper_any._lifecycle_lock:
        wrapper_any._open_order_mapping.clear()

    client_any.reqOpenOrders()

    try:
        orders = await asyncio.wait_for(wrapper_any._open_orders_done, timeout=timeout)
    except TimeoutError:
        log.warning("broker_gateway.fetch_open_orders.timeout")
        orders = list(wrapper_any._open_orders)

    orders = _dedupe_broker_orders(orders)
    accepted = _rebuild_lifecycle_maps(
        orders=orders,
        wrapper=wrapper_any,
        con_id_to_instrument=con_id_to_instrument,
        submitted=submitted,
        execution_policy=execution_policy_any,
    )
    _emit_orphan_events(
        accepted=accepted,
        submitted=submitted,
        wrapper=wrapper_any,
        orphan_ttl_minutes=orphan_ttl_minutes,
    )
    return accepted


def _dedupe_broker_orders(orders: list[BrokerOrder]) -> list[BrokerOrder]:
    """Keep the latest openOrder row per broker_order_id."""
    seen: dict[str, BrokerOrder] = {}
    for broker_order in orders:
        if broker_order.broker_order_id:
            seen[broker_order.broker_order_id] = broker_order
    return list(seen.values())


def _rebuild_lifecycle_maps(
    *,
    orders: list[BrokerOrder],
    wrapper: object,
    con_id_to_instrument: dict[int, uuid.UUID],
    submitted: dict[uuid.UUID, BrokerAck],
    execution_policy: object,
) -> list[BrokerOrder]:
    """Rebuild IB order/instrument maps from staged openOrder callback metadata."""
    wrapper_any = cast("Any", wrapper)
    execution_policy_any = cast("Any", execution_policy)
    with wrapper_any._lifecycle_lock:
        staging = dict(wrapper_any._open_order_mapping)

    accepted: list[BrokerOrder] = []
    for broker_order in orders:
        if not broker_order.broker_order_id:
            continue
        ib_order_id = int(broker_order.broker_order_id)
        ref, con_id = staging.get(ib_order_id, ("", 0))
        internal_id = _parse_order_ref(ref)
        instrument_id = con_id_to_instrument.get(con_id) if con_id else None

        if internal_id is None or instrument_id is None:
            _handle_lost_mapping(
                ib_order_id=ib_order_id,
                order_ref=ref,
                con_id=con_id,
                execution_policy=execution_policy_any,
            )
            continue

        with wrapper_any._lifecycle_lock:
            wrapper_any._ib_to_internal[ib_order_id] = internal_id
            wrapper_any._ib_to_instrument[ib_order_id] = instrument_id
        submitted[broker_order.order_id] = BrokerAck(
            order_id=broker_order.order_id,
            broker_order_id=broker_order.broker_order_id,
            acknowledged_at=broker_order.last_updated_at,
        )
        accepted.append(broker_order)

    return accepted


def _parse_order_ref(ref: str) -> uuid.UUID | None:
    if not ref:
        return None
    try:
        return uuid.UUID(ref)
    except ValueError:
        return None


def _handle_lost_mapping(
    *,
    ib_order_id: int,
    order_ref: str,
    con_id: int,
    execution_policy: object,
) -> None:
    execution_policy = cast("Any", execution_policy)
    reason = (
        "fetch_open_orders: unmapped order on reconnect - "
        f"ib_order_id={ib_order_id} orderRef='{order_ref}' conId={con_id}"
    )
    log.error(
        "broker_gateway.lifecycle.mapping_lost",
        ib_order_id=ib_order_id,
        order_ref=order_ref,
        con_id=con_id,
        reason=reason,
    )
    if execution_policy is not None and hasattr(execution_policy, "activate_kill_switch"):
        try:
            execution_policy.activate_kill_switch(reason, activated_by="session_supervisor")
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "broker_gateway.lifecycle.mapping_lost.kill_switch_failed",
                error=str(exc),
            )


def _emit_orphan_events(
    *,
    accepted: list[BrokerOrder],
    submitted: dict[uuid.UUID, BrokerAck],
    wrapper: object,
    orphan_ttl_minutes: int,
) -> None:
    """Remove stale submitted orders absent from broker open orders."""
    wrapper_any = cast("Any", wrapper)
    if orphan_ttl_minutes <= 0:
        return
    broker_order_ids = {order.broker_order_id for order in accepted if order.broker_order_id}
    cutoff = datetime.now(tz=UTC)
    ttl = timedelta(minutes=orphan_ttl_minutes)
    for order_id, ack in list(submitted.items()):
        if ack.broker_order_id in broker_order_ids:
            continue
        if (cutoff - ack.acknowledged_at) < ttl:
            continue
        log.warning(
            "broker_gateway.orphan_order_detected",
            order_id=str(order_id),
            broker_order_id=ack.broker_order_id,
            acknowledged_at=str(ack.acknowledged_at),
            ttl_minutes=orphan_ttl_minutes,
        )
        with wrapper_any._lifecycle_lock:
            wrapper_any._lifecycle_queue.append(
                BrokerOrphanDetected(
                    order_id=order_id,
                    broker_order_id=ack.broker_order_id or "",
                    acknowledged_at=ack.acknowledged_at,
                    occurred_at=cutoff,
                )
            )
        del submitted[order_id]
