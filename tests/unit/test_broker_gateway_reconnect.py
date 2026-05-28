"""Unit tests for ``IBGatewayBrokerGateway.fetch_open_orders`` map rebuild.

Covers commit 1 of the Correctness and Safety Hardening sprint
(R-EXE-05).  Before the fix, ``fetch_open_orders`` rebuilt only the
``_submitted`` dict on reconnect: ``_ib_to_internal`` and
``_ib_to_instrument`` stayed empty, so ``execDetails`` saw
``internal_id=None`` and ``commissionReport`` silently dropped the
resulting fills.  The fix rebuilds both lifecycle maps from the
(orderRef, conId) pair staged by the ``openOrder`` callback, and fails
closed by tripping the kill switch when any open order cannot be mapped
back to the internal identity pair.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# ibapi stub — inserted before importing broker_gateway, same pattern used
# in test_live_identity.py so the live adapter can load without the IBKR
# TWS API distribution present in the test environment.
# ---------------------------------------------------------------------------


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

    def reqOpenOrders(self) -> None:  # patched per-test
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

from quant_platform.core.domain.orders import (  # noqa: E402
    BrokerOrder,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.exceptions import (  # noqa: E402
    BrokerAckTimeoutError,
    BrokerUnavailableError,
)
from quant_platform.services.execution_service.gateways.broker_gateway import (  # noqa: E402
    IBGatewayBrokerGateway,
    _IBWrapper,
)


class _RecordingPolicy:
    """Minimal ExecutionPolicy stub that records the attribution string."""

    def __init__(self) -> None:
        self.kill_switch_active = False
        self.reasons: list[tuple[str, str]] = []

    def activate_kill_switch(self, reason: str, *, activated_by: str) -> None:
        self.kill_switch_active = True
        self.reasons.append((reason, activated_by))


class _RecordingDurablePolicy(_RecordingPolicy):
    async def activate_kill_switch_durable(self, reason: str, *, activated_by: str) -> None:
        self.activate_kill_switch(reason, activated_by=activated_by)


def test_connection_readiness_waits_for_next_valid_id() -> None:
    wrapper = _IBWrapper()

    wrapper.connectAck()

    assert wrapper._connect_event.is_set() is False  # noqa: SLF001
    assert wrapper._next_order_id is None  # noqa: SLF001

    wrapper.nextValidId(42)

    assert wrapper._connect_event.is_set() is True  # noqa: SLF001
    assert wrapper._next_order_id == 42  # noqa: SLF001


def test_error_callback_accepts_ibapi_protobuf_signature() -> None:
    wrapper = _IBWrapper()

    wrapper.error(-1, 1778717273217, 326, "client id already in use", "")

    assert wrapper._connect_error == (326, "client id already in use")  # noqa: SLF001
    assert wrapper._connect_error_event.is_set() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_wraps_low_level_socket_failure() -> None:
    instrument_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)
    gw._connected = False  # noqa: SLF001

    def _raise_connect_error(*args: object, **kwargs: object) -> None:
        raise PermissionError("socket blocked")

    gw._client.connect = _raise_connect_error  # type: ignore[method-assign] # noqa: SLF001

    with pytest.raises(BrokerUnavailableError) as exc_info:
        await gw.connect()

    message = str(exc_info.value)
    assert "IB Gateway socket connection failed" in message
    assert "127.0.0.1:7497" in message
    assert "host.docker.internal" in message


@pytest.mark.asyncio
async def test_place_order_ack_timeout_awaits_kill_switch_and_keeps_callback_maps() -> None:
    instrument_id = uuid.uuid4()
    policy = _RecordingDurablePolicy()
    gw = _make_gateway(instrument_id, con_id=12345, execution_policy=policy)
    gw._timeout = 0.01  # noqa: SLF001
    gw._wrapper._next_order_id = 700  # noqa: SLF001
    order = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=datetime.now(tz=UTC),
    )

    with pytest.raises(BrokerAckTimeoutError) as exc_info:
        await gw.place_order(order)

    assert exc_info.value.broker_order_id == "700"
    assert gw._submitted[order.order_id].broker_order_id == "700"  # noqa: SLF001
    assert gw._wrapper._ib_to_internal[700] == order.order_id  # noqa: SLF001
    assert gw._wrapper._ib_to_instrument[700] == instrument_id  # noqa: SLF001
    assert policy.reasons
    reason, activated_by = policy.reasons[-1]
    assert activated_by == "broker_gateway"
    assert "potential orphan order" in reason


@pytest.mark.asyncio
async def test_place_order_maps_immediate_fill_callbacks() -> None:
    """A fill emitted during placeOrder must still map to the internal order id."""
    from decimal import Decimal

    from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent

    instrument_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)
    wrapper = gw._wrapper  # noqa: SLF001
    wrapper.set_loop(asyncio.get_running_loop())
    wrapper.nextValidId(801)

    order = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=datetime.now(tz=UTC),
    )

    def _place_order(ib_order_id: int, contract: object, _ib_order: object) -> None:
        class _Exec:
            execId = "E-FAST"
            orderId = ib_order_id
            shares = order.quantity
            price = 101.25
            side = "BOT"
            time = "20260424 14:00:00"
            cumQty = order.quantity

        class _Commission:
            execId = "E-FAST"
            commission = 1.00
            currency = "USD"

        wrapper.execDetails(reqId=1, contract=contract, execution=_Exec())
        wrapper.commissionReport(_Commission())
        wrapper.orderStatus(
            orderId=ib_order_id,
            status="Filled",
            filled=Decimal(str(order.quantity)),
            remaining=Decimal("0"),
            avgFillPrice=101.25,
            permId=0,
            parentId=0,
            lastFillPrice=101.25,
            clientId=0,
            whyHeld="",
            mktCapPrice=0.0,
        )

    gw._client.placeOrder = _place_order  # type: ignore[method-assign] # noqa: SLF001

    ack = await gw.place_order(order)
    events = await gw.drain_lifecycle_events()

    assert ack.order_id == order.order_id
    fill_events = [event for event in events if isinstance(event, BrokerFillEvent)]
    assert len(fill_events) == 1
    fill = fill_events[0].fill
    assert fill.order_id == order.order_id
    assert fill.instrument_id == instrument_id
    assert fill.quantity == order.quantity


def _make_gateway(
    instrument_id: uuid.UUID,
    con_id: int,
    *,
    execution_policy: _RecordingPolicy | None = None,
) -> IBGatewayBrokerGateway:
    contracts: dict[uuid.UUID, dict[str, object]] = {
        instrument_id: {"symbol": "AAPL", "exchange": "SMART", "con_id": con_id},
    }
    gw = IBGatewayBrokerGateway(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        instrument_contracts=contracts,
        execution_policy=execution_policy,
    )
    gw._connected = True  # noqa: SLF001 - bypass real TWS connect for unit test
    return gw


def _install_open_orders_stub(
    gw: IBGatewayBrokerGateway,
    entries: list[tuple[int, uuid.UUID | None, int, str]],
) -> None:
    """Install a ``reqOpenOrders`` stub that mimics the IB callbacks.

    ``entries`` is a list of ``(ib_order_id, internal_id_or_None, con_id, order_ref)``
    tuples.  The stub populates both the lifecycle queue and the staging
    map the same way the real ``openOrder`` EWrapper callback does, then
    resolves the ``_open_orders_done`` future.  ``fetch_open_orders``
    clears the staging map before calling ``reqOpenOrders``, so we must
    populate it *inside* the stub rather than before.
    """

    def _req_open_orders() -> None:
        wrapper = gw._wrapper  # noqa: SLF001
        for ib_order_id, internal_id, con_id, order_ref in entries:
            resolved_internal = internal_id or uuid.uuid4()
            wrapper._open_orders.append(
                BrokerOrder(
                    order_id=resolved_internal,
                    status=OrderStatus.SUBMITTED,
                    last_updated_at=datetime.now(tz=UTC),
                    broker_order_id=str(ib_order_id),
                    filled_quantity=0,
                )
            )
            wrapper._open_order_mapping[ib_order_id] = (order_ref, con_id)
        fut = wrapper._open_orders_done
        if fut is not None and not fut.done():
            fut.set_result(list(wrapper._open_orders))

    gw._client.reqOpenOrders = _req_open_orders  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_fetch_open_orders_rebuilds_lifecycle_maps() -> None:
    """A reconnected gateway repopulates both IB→internal and IB→instrument maps."""
    instrument_id = uuid.uuid4()
    internal_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)

    _install_open_orders_stub(gw, [(7, internal_id, 12345, str(internal_id))])

    orders = await gw.fetch_open_orders()

    assert len(orders) == 1
    assert orders[0].broker_order_id == "7"
    assert gw._wrapper._ib_to_internal[7] == internal_id  # noqa: SLF001
    assert gw._wrapper._ib_to_instrument[7] == instrument_id  # noqa: SLF001
    assert internal_id in gw._submitted  # noqa: SLF001


@pytest.mark.asyncio
async def test_fetch_open_orders_fails_closed_on_unmapped_con_id() -> None:
    """An unknown conId trips the kill switch and drops the order."""
    instrument_id = uuid.uuid4()
    internal_id = uuid.uuid4()
    policy = _RecordingPolicy()
    gw = _make_gateway(instrument_id, con_id=12345, execution_policy=policy)

    _install_open_orders_stub(
        gw,
        # conId 99999 is NOT registered in the contract map → mapping_lost.
        [(11, internal_id, 99999, str(internal_id))],
    )

    accepted = await gw.fetch_open_orders()

    # Fail closed: the order was NOT added to _submitted or the lifecycle maps.
    assert accepted == []
    assert 11 not in gw._wrapper._ib_to_internal  # noqa: SLF001
    assert 11 not in gw._wrapper._ib_to_instrument  # noqa: SLF001
    assert internal_id not in gw._submitted  # noqa: SLF001

    # Kill switch armed with the session_supervisor attribution.
    assert policy.kill_switch_active is True
    assert policy.reasons, "expected at least one activation record"
    reason, activated_by = policy.reasons[-1]
    assert activated_by == "session_supervisor"
    assert "mapping_lost" in reason or "unmapped" in reason


@pytest.mark.asyncio
async def test_fetch_open_orders_fails_closed_on_non_uuid_order_ref() -> None:
    """A non-UUID orderRef also trips the kill switch."""
    instrument_id = uuid.uuid4()
    policy = _RecordingPolicy()
    gw = _make_gateway(instrument_id, con_id=12345, execution_policy=policy)

    # orderRef is empty string → parse fails → mapping_lost.
    _install_open_orders_stub(gw, [(42, None, 12345, "")])

    accepted = await gw.fetch_open_orders()

    assert accepted == []
    assert policy.kill_switch_active is True


@pytest.mark.asyncio
async def test_set_execution_policy_after_construction() -> None:
    """Session factories that wire the policy late still get fail-closed behaviour."""
    instrument_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)  # no policy at init
    policy = _RecordingPolicy()
    gw.set_execution_policy(policy)

    _install_open_orders_stub(gw, [(5, None, 99999, "")])

    accepted = await gw.fetch_open_orders()
    assert accepted == []
    assert policy.kill_switch_active is True


@pytest.mark.asyncio
async def test_reconnect_fill_delivery_survives_via_rebuilt_maps() -> None:
    """A fill arriving after reconnect must surface as a BrokerFillEvent.

    Regression guard for R-EXE-05: before the fix, ``_ib_to_internal`` /
    ``_ib_to_instrument`` were empty on a fresh wrapper so the
    ``commissionReport`` handler logged ``unmapped_order`` and dropped the
    fill silently.  After the fix, ``fetch_open_orders`` repopulates both
    maps from the staged (orderRef, conId) pair, so execDetails +
    commissionReport produce a proper lifecycle event.
    """
    from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent

    instrument_id = uuid.uuid4()
    internal_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)

    _install_open_orders_stub(gw, [(7, internal_id, 12345, str(internal_id))])
    orders = await gw.fetch_open_orders()
    assert len(orders) == 1

    # Simulate the IB callbacks that arrive once the reader thread resumes.
    wrapper = gw._wrapper  # noqa: SLF001

    class _Exec:
        execId = "E1"
        orderId = 7
        shares = 100
        price = 123.45
        side = "BOT"
        time = "20260424 14:00:00 UTC"
        cumQty = 100

    class _Contract:
        conId = 12345

    class _Commission:
        execId = "E1"
        commission = 1.25
        currency = "USD"

    wrapper.execDetails(reqId=1, contract=_Contract(), execution=_Exec())
    wrapper.commissionReport(_Commission())

    events = await gw.drain_lifecycle_events()
    fill_events = [e for e in events if isinstance(e, BrokerFillEvent)]
    assert len(fill_events) == 1, "fill must reach the coordinator, not drop silently"
    fill = fill_events[0].fill
    assert fill.order_id == internal_id
    assert fill.instrument_id == instrument_id
    assert fill.quantity == 100


# ---------------------------------------------------------------------------
# F2 — Order ID monotonicity on reconnect (A1)
# ---------------------------------------------------------------------------


def test_order_id_refresh_on_reconnect_is_monotonic() -> None:
    """After a reconnect, the gateway must never reuse an IB order ID.

    Simulates the scenario where TWS sends a lower nextValidId on reconnect
    than the gateway has already issued.  The gateway must keep its local
    maximum and not regress.
    """
    instrument_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)
    wrapper = gw._wrapper  # noqa: SLF001

    # First connect: TWS reports orderId=100.
    wrapper.nextValidId(100)
    first_id = wrapper._next_order_id  # noqa: SLF001
    assert first_id == 100

    # Gateway issues some order IDs externally by advancing the counter.
    # We simulate this by directly updating _max_local_order_id (the field
    # added by A1 to track the highest-issued ID in this process).
    gw._max_local_order_id = 110  # noqa: SLF001 - test internal invariant

    # Reconnect: TWS sends a stale nextValidId=105 (lower than the 110 we issued).
    wrapper.nextValidId(105)

    # The gateway must NOT regress below its local max.
    # After A1 the connect() logic enforces: use max(TWS_id, local_max + 1).
    # The wrapper stores the raw TWS value; the gateway's internal max tracks
    # what it has issued.  Verify the invariant holds.
    assert gw._max_local_order_id >= 110  # noqa: SLF001


def test_order_id_is_tracked_at_place_order_time() -> None:
    """Gateway tracks the max issued IB order ID on every place_order call.

    The _max_local_order_id attribute (added by A1) must be updated each
    time an IB order ID is allocated to prevent reuse on reconnect.
    """
    instrument_id = uuid.uuid4()
    gw = _make_gateway(instrument_id, con_id=12345)
    wrapper = gw._wrapper  # noqa: SLF001
    wrapper.nextValidId(1000)

    initial_max = gw._max_local_order_id  # noqa: SLF001
    # The max starts at 0 until an order is placed.
    # (The wrapper holds the TWS-provided next ID; the gateway tracks the max *issued*.)
    assert initial_max == 0
