"""Tests for the durable kill switch (Phase 3.2).

Covers:
- ``InMemoryKillSwitchStore`` round-trip.
- ``OrderThrottle.hydrate_kill_switch`` restoring state without
  reactivating the store write path.
- ``OrderThrottle.activate_kill_switch`` + ``clear_kill_switch``
  persisting through the store when a running loop is available.
- ``hydrate_session_state`` no-op when no store is wired.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from quant_platform.config import PlatformSettings, StorageSettings
from quant_platform.services.execution_service.orders.order_throttle import OrderThrottle
from quant_platform.services.execution_service.stores.kill_switch_store import (
    InMemoryKillSwitchStore,
    KillSwitchState,
)
from quant_platform.session import hydrate_session_state


class _FakeClock:
    """Deterministic clock sufficient for throttle + store write tests."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


class _FailingStore:
    async def get(self) -> object:
        raise RuntimeError("store unavailable")


class _FailingCoordinator:
    async def hydrate(self) -> None:
        raise RuntimeError("coordinator unavailable")


def _session_stub(*, postgres_dsn: str, store: object | None, coordinator: object | None) -> object:
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn=postgres_dsn),
    )
    return SimpleNamespace(
        settings=settings,
        kill_switch_store=store,
        execution_policy=OrderThrottle(_FakeClock(datetime(2026, 4, 23, tzinfo=UTC))),
        coordinator=coordinator,
        _state_hydrated=False,
    )


@pytest.mark.asyncio
async def test_in_memory_store_round_trip() -> None:
    store = InMemoryKillSwitchStore()
    state = await store.get()
    assert state.active is False and state.reason is None

    ts = datetime(2026, 4, 23, tzinfo=UTC)
    await store.activate(reason="drift", activated_by="guard", as_of=ts)
    state = await store.get()
    assert state == KillSwitchState(
        active=True, reason="drift", activated_at=ts, activated_by="guard"
    )

    await store.clear(operator_id="alice", as_of=ts)
    state = await store.get()
    assert state.active is False
    assert state.activated_by == "alice"


@pytest.mark.asyncio
async def test_throttle_activate_writes_through_store() -> None:
    """``activate_kill_switch`` persists through the store when looped."""
    store = InMemoryKillSwitchStore()
    clock = _FakeClock(datetime(2026, 4, 23, tzinfo=UTC))
    throttle = OrderThrottle(clock, kill_switch_store=store)

    throttle.activate_kill_switch("integration test", activated_by="cash_drift_guard")
    # The throttle schedules store.activate() as a background task; let
    # the loop drain before asserting.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    state = await store.get()
    assert state.active is True
    assert state.reason == "integration test"
    assert state.activated_by == "cash_drift_guard"

    throttle.clear_kill_switch("ops")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    state = await store.get()
    assert state.active is False
    assert state.activated_by == "ops"


def test_throttle_hydrate_does_not_write_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """``hydrate_kill_switch`` restores state without scheduling a write."""
    store = InMemoryKillSwitchStore()
    clock = _FakeClock(datetime(2026, 4, 23, tzinfo=UTC))
    throttle = OrderThrottle(clock, kill_switch_store=store)

    throttle.hydrate_kill_switch(active=True, reason="post-restart")
    assert throttle.kill_switch_active is True
    # No async loop running, so nothing is scheduled.  Store remains
    # fresh.
    assert store._state.active is False  # noqa: SLF001 - test invariant


def test_throttle_without_store_uses_in_memory_state() -> None:
    """Omitting the store keeps the in-memory-only behaviour."""
    clock = _FakeClock(datetime(2026, 4, 23, tzinfo=UTC))
    throttle = OrderThrottle(clock)
    throttle.activate_kill_switch("no-store", activated_by="unit_test")
    assert throttle.kill_switch_active is True
    throttle.clear_kill_switch("ops")
    assert throttle.kill_switch_active is False


# ---------------------------------------------------------------------------
# R-OBS-04 attribution tests: every real caller must persist the correct
# ``activated_by`` value so the kill_switch_state row truthfully records who
# halted the system.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activated_by",
    [
        "cash_drift_guard",
        "distributed_lock",
        "session_supervisor",
        "reconciliation",
    ],
)
@pytest.mark.asyncio
async def test_activated_by_roundtrips_for_each_actor(activated_by: str) -> None:
    """Each known actor string lands verbatim in the durable store."""
    store = InMemoryKillSwitchStore()
    clock = _FakeClock(datetime(2026, 4, 24, tzinfo=UTC))
    throttle = OrderThrottle(clock, kill_switch_store=store)

    throttle.activate_kill_switch(f"triggered by {activated_by}", activated_by=activated_by)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    state = await store.get()
    assert state.active is True
    assert state.activated_by == activated_by
    assert state.reason == f"triggered by {activated_by}"


def test_activate_requires_activated_by_keyword() -> None:
    """Callers must name the actor — positional / missing is a TypeError."""
    clock = _FakeClock(datetime(2026, 4, 24, tzinfo=UTC))
    throttle = OrderThrottle(clock)

    with pytest.raises(TypeError):
        throttle.activate_kill_switch("positional-only")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_hydrate_session_state_fails_closed_on_durable_store_failure() -> None:
    session = _session_stub(
        postgres_dsn="postgresql+psycopg://quant:pw@db/quant_platform",
        store=_FailingStore(),
        coordinator=None,
    )

    with pytest.raises(RuntimeError, match="kill-switch"):
        await hydrate_session_state(session)  # type: ignore[arg-type]

    assert session._state_hydrated is False


@pytest.mark.asyncio
async def test_hydrate_session_state_fails_closed_on_durable_coordinator_failure() -> None:
    session = _session_stub(
        postgres_dsn="postgresql+psycopg://quant:pw@db/quant_platform",
        store=None,
        coordinator=_FailingCoordinator(),
    )

    with pytest.raises(RuntimeError, match="coordinator"):
        await hydrate_session_state(session)  # type: ignore[arg-type]

    assert session._state_hydrated is False


@pytest.mark.asyncio
async def test_hydrate_session_state_keeps_in_memory_fail_open_behavior() -> None:
    session = _session_stub(
        postgres_dsn="",
        store=_FailingStore(),
        coordinator=_FailingCoordinator(),
    )

    await hydrate_session_state(session)  # type: ignore[arg-type]

    assert session._state_hydrated is True
