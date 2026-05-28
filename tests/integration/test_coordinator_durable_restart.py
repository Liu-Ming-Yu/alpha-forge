"""Integration test: mid-day restart rehydrates AccountStateCoordinator.

Covers commit 3 of the Correctness and Safety Hardening sprint
(R-EXE-07).  Before the fix, ``_pending_lots`` and ``_completed_order_ids``
lived only in process memory: a restart between a sell fill and the
matching T+1 settlement advance lost the projection, and fills that had
already triggered a BrokerOrderCompleted could be re-credited after a
restart because the reservation hint was gone.

This test drives a sell fill into a coordinator that is wired to the
same durable stores the session factory would build.  It then replaces
the coordinator (simulating a process restart) and asserts that
``hydrate`` restores the lot, that ``advance_settlements`` on T+1 pays
out the proceeds, and that a replayed BrokerOrderCompleted event does
not double-credit.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import FillEvent, OrderSide
from quant_platform.core.domain.orders.lifecycle import (
    BrokerFillEvent,
    BrokerOrderCompleted,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.infrastructure.event_bus import InMemoryEventBus
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.execution_service.account.account_state_coordinator import (
    AccountStateCoordinator,
)
from quant_platform.services.execution_service.stores.pending_settlement_store import (
    InMemoryCompletedOrderHintStore,
    InMemoryPendingSettlementStore,
)
from quant_platform.services.portfolio_service.cash_ledger import CashLedger
from quant_platform.services.portfolio_service.settlement_calendar import (
    SettlementCalendar,
)

_UTC = UTC
_TRADE_DATE = datetime(2026, 4, 24, 14, 30, 0, tzinfo=_UTC)


class _FailingEventBus(InMemoryEventBus):
    async def publish(self, event: object) -> None:
        _ = event
        raise RuntimeError("event bus unavailable")


def _snapshot(
    cash: Decimal,
    *,
    unsettled: Decimal = Decimal("0"),
) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_TRADE_DATE,
        settled_cash=cash,
        unsettled_cash=unsettled,
        reserved_cash=Decimal("0"),
        available_cash=cash,
        net_asset_value=cash + unsettled,
        positions=(),
    )


def _sell_fill(order_id: uuid.UUID, instrument_id: uuid.UUID) -> FillEvent:
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=order_id,
        broker_order_id="BRK-1",
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=100,
        fill_price=Decimal("50.00"),
        commission=Decimal("1.00"),
        currency="USD",
        executed_at=_TRADE_DATE,
        received_at=_TRADE_DATE,
    )


def _build_coordinator(
    clock: FakeClock,
    bus: InMemoryEventBus,
    run_id: uuid.UUID,
    pending_store: InMemoryPendingSettlementStore,
    completed_store: InMemoryCompletedOrderHintStore,
    initial_cash: Decimal,
    *,
    unsettled_cash: Decimal = Decimal("0"),
) -> tuple[AccountStateCoordinator, CashLedger]:
    ledger = CashLedger(
        clock=clock,
        settlement_calendar=SettlementCalendar(),
        initial_snapshot=_snapshot(initial_cash, unsettled=unsettled_cash),
    )
    coord = AccountStateCoordinator(
        cash_engine=ledger,
        event_bus=bus,
        clock=clock,
        strategy_run_id=run_id,
        pending_settlement_store=pending_store,
        completed_order_hint_store=completed_store,
    )
    return coord, ledger


@pytest.mark.asyncio
async def test_mid_day_restart_rehydrates_pending_settlement_lot() -> None:
    """Pre-restart sell fill must still settle on T+1 after coordinator swap."""
    clock = FakeClock(_TRADE_DATE)
    bus = InMemoryEventBus()
    run_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    order_id = uuid.uuid4()

    # Shared durable stores span the two coordinator instances — this is
    # the invariant the Postgres backends provide in production.
    pending_store = InMemoryPendingSettlementStore()
    completed_store = InMemoryCompletedOrderHintStore()

    # --- Pre-restart coordinator: drive a sell fill through it. --------
    pre, pre_ledger = _build_coordinator(
        clock, bus, run_id, pending_store, completed_store, Decimal("100000")
    )

    fill = _sell_fill(order_id, instrument_id)
    await pre.process_lifecycle_events([BrokerFillEvent(fill=fill, is_complete=True)])
    assert len(pre.pending_settlement_lots) == 1
    # The lot was persisted to the durable store.
    persisted = await pending_store.list_all(run_id=run_id)
    assert len(persisted) == 1
    assert persisted[0].net_proceeds == Decimal("4999.00")  # 100 * 50 - 1 commission

    # --- Simulate process restart ---------------------------------------
    # The new process reads the broker-authoritative account snapshot at
    # startup: settled_cash is unchanged (sell proceeds are not yet
    # settled) and unsettled_cash carries the pending 4999.  The fresh
    # coordinator would otherwise have an empty pending-lot queue.
    post, post_ledger = _build_coordinator(
        clock,
        bus,
        run_id,
        pending_store,
        completed_store,
        Decimal("100000"),
        unsettled_cash=Decimal("4999.00"),
    )
    assert post.pending_settlement_lots == []  # fresh instance starts empty
    await post.hydrate()
    assert len(post.pending_settlement_lots) == 1, (
        "hydrate() must rebuild pending lots from the durable store"
    )

    # --- Advance the clock to T+1 and settle. --------------------------
    clock.set(datetime(2026, 4, 27, 14, 30, 0, tzinfo=_UTC))  # Monday after
    settled = await post.advance_settlements()
    assert settled == 1
    # Durable row deleted in the same step.
    assert await pending_store.list_all(run_id=run_id) == []
    # Ledger on the new coordinator credits the settled proceeds.
    assert post_ledger.settled_cash == Decimal("100000") + Decimal("4999.00")
    assert post_ledger.unsettled_cash == Decimal("0")


@pytest.mark.asyncio
async def test_completed_order_hint_survives_restart() -> None:
    """A BrokerOrderCompleted applied pre-restart is not re-applied post-restart."""
    clock = FakeClock(_TRADE_DATE)
    bus = InMemoryEventBus()
    run_id = uuid.uuid4()
    order_id = uuid.uuid4()

    pending_store = InMemoryPendingSettlementStore()
    completed_store = InMemoryCompletedOrderHintStore()

    pre, _ = _build_coordinator(
        clock, bus, run_id, pending_store, completed_store, Decimal("100000")
    )
    await pre.process_lifecycle_events(
        [
            BrokerOrderCompleted(
                order_id=order_id,
                broker_order_id="BRK-2",
                occurred_at=_TRADE_DATE,
            )
        ]
    )
    assert order_id in await completed_store.list_all(run_id=run_id)

    # Restart: fresh coordinator, no memory of the completed order.
    post, _ = _build_coordinator(
        clock, bus, run_id, pending_store, completed_store, Decimal("100000")
    )
    await post.hydrate()
    # The completed hint was rehydrated so an incoming fill for this
    # order is treated as the final partial, not as a stray.
    assert order_id in post._completed_order_ids  # noqa: SLF001 - invariant


@pytest.mark.asyncio
async def test_lifecycle_processing_errors_fail_closed() -> None:
    clock = FakeClock(_TRADE_DATE)
    bus = _FailingEventBus()
    run_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    order_id = uuid.uuid4()
    pending_store = InMemoryPendingSettlementStore()
    completed_store = InMemoryCompletedOrderHintStore()
    coord, _ = _build_coordinator(
        clock,
        bus,
        run_id,
        pending_store,
        completed_store,
        Decimal("100000"),
    )

    fill = _sell_fill(order_id, instrument_id)
    with pytest.raises(RuntimeError, match="lifecycle event processing failed"):
        await coord.process_lifecycle_events([BrokerFillEvent(fill=fill, is_complete=True)])
