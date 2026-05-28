"""Tests for strategy-cycle step helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.core.domain.orders import FillEvent, OrderSide
from quant_platform.core.domain.orders.lifecycle import BrokerFillEvent, BrokerLifecycleEvent
from quant_platform.engines.session.cycle_steps import drain_cycle_lifecycle_events

_NOW = datetime(2026, 5, 18, 15, 0, tzinfo=UTC)


def _fill() -> FillEvent:
    return FillEvent(
        fill_id=uuid.uuid4(),
        order_id=uuid.uuid4(),
        broker_order_id="ib-1",
        broker_execution_id="exec-1",
        instrument_id=uuid.uuid4(),
        side=OrderSide.BUY,
        quantity=10,
        fill_price=Decimal("100"),
        commission=Decimal("1"),
        currency="USD",
        executed_at=_NOW,
        received_at=_NOW,
    )


class _DelayedLifecycleFeed:
    def __init__(self, batches: list[list[BrokerLifecycleEvent]]) -> None:
        self._batches = list(batches)
        self.calls = 0

    async def drain_lifecycle_events(self) -> list[BrokerLifecycleEvent]:
        self.calls += 1
        if self._batches:
            return self._batches.pop(0)
        return []


class _RecordingCoordinator:
    def __init__(self) -> None:
        self.batches: list[list[BrokerLifecycleEvent]] = []

    async def process_lifecycle_events(
        self,
        events: list[BrokerLifecycleEvent],
    ) -> None:
        self.batches.append(events)


@pytest.mark.asyncio
async def test_post_submit_drain_polls_for_late_lifecycle_fill() -> None:
    fill_event = BrokerFillEvent(fill=_fill(), is_complete=False)
    feed = _DelayedLifecycleFeed([[], [fill_event]])
    coordinator = _RecordingCoordinator()
    session = SimpleNamespace(lifecycle_feed=feed, coordinator=coordinator)

    fills = await drain_cycle_lifecycle_events(
        session=session,  # type: ignore[arg-type]
        engine_name="test_engine",
        wait_timeout_seconds=0.05,
        poll_interval_seconds=0.01,
    )

    assert fills == [fill_event.fill]
    assert coordinator.batches == [[fill_event]]
    assert feed.calls >= 2


@pytest.mark.asyncio
async def test_zero_timeout_drain_does_not_wait_for_late_lifecycle_fill() -> None:
    fill_event = BrokerFillEvent(fill=_fill(), is_complete=False)
    feed = _DelayedLifecycleFeed([[], [fill_event]])
    coordinator = _RecordingCoordinator()
    session = SimpleNamespace(lifecycle_feed=feed, coordinator=coordinator)

    fills = await drain_cycle_lifecycle_events(
        session=session,  # type: ignore[arg-type]
        engine_name="test_engine",
        wait_timeout_seconds=0.0,
        poll_interval_seconds=0.01,
    )

    assert fills == []
    assert coordinator.batches == []
    assert feed.calls == 1
