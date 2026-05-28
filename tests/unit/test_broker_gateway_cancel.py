"""Unit tests for the cancel-future resolution path on orderStatus=Cancelled.

Covers commit 2 of the Correctness and Safety Hardening sprint
(R-EXE-06).  Before the fix, ``_cancel_futures[orderId]`` was resolved
only in the ``error()`` handler for IB codes 201/202, which is an error
path.  Routine cancels succeed through ``orderStatus(status=Cancelled)``
without any error() delivery, so the future remained pending until the
caller's ack timeout fired as ``BrokerUnavailableError`` — making every
successful cancel look like a broker failure.

The fix: resolve the cancel future inside ``orderStatus`` for
``status in {Cancelled, ApiCancelled}`` and emit
``BrokerOrderCancelled`` once per order.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeEWrapper:
    def __init__(self) -> None:
        pass


class _FakeEClient:
    def __init__(self, wrapper: Any) -> None:
        self._wrapper = wrapper

    def connect(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def reqOpenOrders(self) -> None:
        pass

    def placeOrder(self, *args: Any, **kwargs: Any) -> None:
        pass


def _install_ibapi_stubs() -> None:
    try:
        __import__("ibapi.client")
        __import__("ibapi.wrapper")
        __import__("ibapi.contract")
        __import__("ibapi.order")
        __import__("ibapi.common")
        return
    except Exception:
        pass

    fake_client_mod = MagicMock()
    fake_client_mod.EClient = _FakeEClient
    fake_wrapper_mod = MagicMock()
    fake_wrapper_mod.EWrapper = _FakeEWrapper
    fake_contract_mod = MagicMock()
    fake_contract_mod.Contract = type(
        "Contract",
        (),
        {
            "conId": 0,
            "symbol": "",
            "exchange": "",
            "currency": "USD",
            "secType": "STK",
            "primaryExchange": "",
            "__init__": lambda self: None,
        },
    )
    for name, mod in [
        ("ibapi", MagicMock()),
        ("ibapi.client", fake_client_mod),
        ("ibapi.wrapper", fake_wrapper_mod),
        ("ibapi.contract", fake_contract_mod),
        ("ibapi.order", MagicMock()),
        ("ibapi.common", MagicMock()),
    ]:
        sys.modules.setdefault(name, mod)


_install_ibapi_stubs()

from quant_platform.core.domain.orders.lifecycle import BrokerOrderCancelled  # noqa: E402
from quant_platform.core.exceptions import BrokerSubmissionError  # noqa: E402
from quant_platform.services.execution_service.gateways.broker_gateway import (  # noqa: E402
    _IBWrapper,
)
from quant_platform.services.execution_service.ib.ib_order_submission import (  # noqa: E402
    cancel_order_and_wait,
)


def _zero() -> Decimal:
    return Decimal("0")


@pytest.mark.asyncio
async def test_order_status_cancelled_resolves_cancel_future() -> None:
    """A happy-path cancel delivered via orderStatus must complete the future.

    Previously only ``error()`` codes 201/202 resolved the cancel future; a
    routine cancel (no error() delivery) timed out as BrokerUnavailableError.
    """
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 42
    internal_id = uuid.uuid4()
    wrapper._ib_to_internal[ib_order_id] = internal_id  # noqa: SLF001

    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper._cancel_futures[ib_order_id] = cancel_future  # noqa: SLF001

    # Deliver a Cancelled orderStatus — no error() call involved.
    wrapper.orderStatus(
        orderId=ib_order_id,
        status="Cancelled",
        filled=_zero(),
        remaining=Decimal("100"),
        avgFillPrice=0.0,
        permId=0,
        parentId=0,
        lastFillPrice=0.0,
        clientId=0,
        whyHeld="",
        mktCapPrice=0.0,
    )

    # _resolve posts through call_soon_threadsafe — yield once.
    await asyncio.sleep(0)

    assert cancel_future.done()
    assert cancel_future.result() is None


@pytest.mark.asyncio
async def test_api_cancelled_status_resolves_cancel_future() -> None:
    """``ApiCancelled`` is treated symmetrically to ``Cancelled``."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 101
    wrapper._ib_to_internal[ib_order_id] = uuid.uuid4()  # noqa: SLF001
    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper._cancel_futures[ib_order_id] = cancel_future  # noqa: SLF001

    wrapper.orderStatus(
        orderId=ib_order_id,
        status="ApiCancelled",
        filled=_zero(),
        remaining=Decimal("50"),
        avgFillPrice=0.0,
        permId=0,
        parentId=0,
        lastFillPrice=0.0,
        clientId=0,
        whyHeld="",
        mktCapPrice=0.0,
    )
    await asyncio.sleep(0)
    assert cancel_future.done()


@pytest.mark.asyncio
async def test_cancel_order_uses_order_cancel_payload_for_current_ibapi() -> None:
    """Current IB API expects an OrderCancel payload, not a raw string."""
    wrapper = type("Wrapper", (), {"_cancel_futures": {}})()

    class _Client:
        def cancelOrder(self, order_id: int, order_cancel: object) -> None:
            assert order_id == 42
            assert hasattr(order_cancel, "manualOrderCancelTime")
            wrapper._cancel_futures[order_id].set_result(None)

    await cancel_order_and_wait(
        client=_Client(),
        wrapper=wrapper,
        timeout=1.0,
        broker_order_id="42",
    )


@pytest.mark.asyncio
async def test_duplicate_cancel_callbacks_do_not_double_complete_future() -> None:
    """IB can deliver both Cancelled status and error 202 for one cancel."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    ib_order_id = 43
    wrapper._ib_to_internal[ib_order_id] = uuid.uuid4()  # noqa: SLF001
    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper._cancel_futures[ib_order_id] = cancel_future  # noqa: SLF001
    try:
        wrapper.orderStatus(
            orderId=ib_order_id,
            status="Cancelled",
            filled=_zero(),
            remaining=Decimal("1"),
            avgFillPrice=0.0,
            permId=0,
            parentId=0,
            lastFillPrice=0.0,
            clientId=0,
            whyHeld="",
            mktCapPrice=0.0,
        )
        wrapper.error(ib_order_id, 202, "Order Canceled - reason:")
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert cancel_future.done()
    assert loop_errors == []


@pytest.mark.asyncio
async def test_cancel_emits_lifecycle_event_exactly_once() -> None:
    """Repeated Cancelled orderStatus deliveries produce exactly one event."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 7
    internal_id = uuid.uuid4()
    wrapper._ib_to_internal[ib_order_id] = internal_id  # noqa: SLF001

    for _ in range(3):
        wrapper.orderStatus(
            orderId=ib_order_id,
            status="Cancelled",
            filled=_zero(),
            remaining=Decimal("100"),
            avgFillPrice=0.0,
            permId=0,
            parentId=0,
            lastFillPrice=0.0,
            clientId=0,
            whyHeld="",
            mktCapPrice=0.0,
        )

    cancel_events = [
        e
        for e in wrapper._lifecycle_queue  # noqa: SLF001
        if isinstance(e, BrokerOrderCancelled)
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].order_id == internal_id


@pytest.mark.asyncio
async def test_non_cancel_status_leaves_cancel_future_pending() -> None:
    """Only Cancelled / ApiCancelled touch the cancel future."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 9
    wrapper._ib_to_internal[ib_order_id] = uuid.uuid4()  # noqa: SLF001
    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper._cancel_futures[ib_order_id] = cancel_future  # noqa: SLF001

    wrapper.orderStatus(
        orderId=ib_order_id,
        status="Filled",
        filled=Decimal("100"),
        remaining=_zero(),
        avgFillPrice=100.0,
        permId=0,
        parentId=0,
        lastFillPrice=100.0,
        clientId=0,
        whyHeld="",
        mktCapPrice=0.0,
    )
    await asyncio.sleep(0)
    assert not cancel_future.done(), "Filled orderStatus must not resolve a pending cancel future"


@pytest.mark.asyncio
async def test_order_error_rejects_pending_order_status_future() -> None:
    """Broker errors tied to an order ID should fail place_order immediately."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 12
    status_future: asyncio.Future[str] = loop.create_future()
    wrapper._order_statuses[ib_order_id] = status_future  # noqa: SLF001

    wrapper.error(
        reqId=ib_order_id,
        errorCode=10268,
        errorString="The 'EtradeOnly' order attribute is not supported.",
    )
    await asyncio.sleep(0)

    assert status_future.done()
    with pytest.raises(BrokerSubmissionError):
        status_future.result()


@pytest.mark.asyncio
async def test_already_cancelled_error_resolves_cancel_future() -> None:
    """A duplicate cancel for an already-cancelled order should not hang."""
    wrapper = _IBWrapper()
    loop = asyncio.get_running_loop()
    wrapper.set_loop(loop)

    ib_order_id = 13
    cancel_future: asyncio.Future[None] = loop.create_future()
    wrapper._cancel_futures[ib_order_id] = cancel_future  # noqa: SLF001

    wrapper.error(
        reqId=ib_order_id,
        errorCode=10148,
        errorString="OrderId 13 that needs to be cancelled cannot be cancelled, state: Cancelled.",
    )
    await asyncio.sleep(0)

    assert cancel_future.done()
    assert cancel_future.result() is None
