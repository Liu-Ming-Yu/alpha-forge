"""IB order submission and cancellation helpers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.core.contracts import BrokerAck
from quant_platform.core.exceptions import (
    BrokerAckTimeoutError,
    BrokerSubmissionError,
    BrokerUnavailableError,
)
from quant_platform.services.execution_service.ib.ib_order_mapper import build_ib_order

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.domain.orders import OrderIntent

log = structlog.get_logger(__name__)


async def submit_order_and_wait_for_ack(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    order: OrderIntent,
    ib_order_id: int,
    contract: object,
    submitted: dict[uuid.UUID, BrokerAck],
    execution_policy: object,
) -> BrokerAck:
    """Submit one IB order and wait for the initial broker status callback."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    ib_order = build_ib_order(order)
    loop = asyncio.get_running_loop()
    status_future: asyncio.Future[str] = loop.create_future()
    wrapper_any._order_statuses[ib_order_id] = status_future

    log.info(
        "broker_gateway.place_order",
        order_id=str(order.order_id),
        ib_order_id=ib_order_id,
    )

    # Populate lifecycle maps before placeOrder(): paper market orders can fill
    # quickly enough that execDetails/commissionReport arrive while placeOrder()
    # is still on the stack.  Without this map, those fills are orphaned and the
    # strategy-cycle summary can show fills=0 even though broker positions moved.
    with wrapper_any._lifecycle_lock:
        wrapper_any._ib_to_internal[ib_order_id] = order.order_id
        # Register canonical instrument_id so commissionReport can attach the
        # correct internal ID to fills without re-deriving it from the broker conId.
        wrapper_any._ib_to_instrument[ib_order_id] = order.instrument_id

    try:
        client_any.placeOrder(ib_order_id, contract, ib_order)
    except Exception:
        # placeOrder() raised before transmitting; clean up the status future
        # and pre-registered lifecycle maps so a failed local call cannot poison
        # future callbacks.
        wrapper_any._order_statuses.pop(ib_order_id, None)
        with wrapper_any._lifecycle_lock:
            wrapper_any._ib_to_internal.pop(ib_order_id, None)
            wrapper_any._ib_to_instrument.pop(ib_order_id, None)
        raise

    try:
        status = await asyncio.wait_for(status_future, timeout=timeout)
    except TimeoutError as exc:
        ack = BrokerAck(
            order_id=order.order_id,
            broker_order_id=str(ib_order_id),
            acknowledged_at=datetime.now(tz=UTC),
        )
        submitted[order.order_id] = ack
        reason = f"place_order timeout - potential orphan order {order.order_id}"
        await _activate_timeout_kill_switch(execution_policy, reason)
        raise BrokerAckTimeoutError(
            f"place_order timed out for order {order.order_id}",
            order_id=order.order_id,
            broker_order_id=str(ib_order_id),
        ) from exc
    finally:
        popped = wrapper_any._order_statuses.pop(ib_order_id, None)
        if popped is not None and not popped.done():
            popped.cancel()

    if status in ("Cancelled", "Inactive"):
        raise BrokerSubmissionError(f"IB rejected order: status={status}")

    ack = BrokerAck(
        order_id=order.order_id,
        broker_order_id=str(ib_order_id),
        acknowledged_at=datetime.now(tz=UTC),
    )
    submitted[order.order_id] = ack
    return ack


async def cancel_order_and_wait(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    broker_order_id: str,
) -> None:
    """Cancel one IB order and wait for the cancel future to resolve."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    ib_order_id = int(broker_order_id)
    loop = asyncio.get_running_loop()
    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper_any._cancel_futures[ib_order_id] = cancel_future

    log.info("broker_gateway.cancel_order", broker_order_id=broker_order_id)
    try:
        client_any.cancelOrder(ib_order_id, _order_cancel_payload())
    except TypeError:
        try:
            # ibapi 9.81 exposes cancelOrder(orderId); intermediate APIs used
            # cancelOrder(orderId, manualOrderCancelTime). Support both so
            # paper cleanup is not blocked by the installed ibapi version.
            client_any.cancelOrder(ib_order_id, "")
        except TypeError:
            client_any.cancelOrder(ib_order_id)

    try:
        await asyncio.wait_for(cancel_future, timeout=timeout)
    except TimeoutError as exc:
        raise BrokerUnavailableError(
            f"cancel_order timed out for broker order {broker_order_id}"
        ) from exc
    finally:
        popped_cancel = wrapper_any._cancel_futures.pop(ib_order_id, None)
        if popped_cancel is not None and not popped_cancel.done():
            popped_cancel.cancel()


def _order_cancel_payload() -> object:
    try:
        from ibapi.order_cancel import OrderCancel

        return OrderCancel()
    except Exception:
        return _FallbackOrderCancel()


class _FallbackOrderCancel:
    """Minimal payload shape accepted by current IB cancelOrder APIs."""

    manualOrderCancelTime = ""  # noqa: N815 - IB API field name


async def _activate_timeout_kill_switch(execution_policy: object, reason: str) -> None:
    execution_policy = cast("Any", execution_policy)
    if execution_policy is None:
        return
    if hasattr(execution_policy, "activate_kill_switch_durable"):
        await execution_policy.activate_kill_switch_durable(
            reason,
            activated_by="broker_gateway",
        )
    elif hasattr(execution_policy, "activate_kill_switch"):
        execution_policy.activate_kill_switch(
            reason,
            activated_by="broker_gateway",
        )
