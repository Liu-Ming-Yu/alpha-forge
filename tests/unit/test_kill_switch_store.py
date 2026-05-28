"""Unit tests for InMemoryKillSwitchStore (Stream 2 hardening)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.services.execution_service.stores.kill_switch_store import (
    InMemoryKillSwitchStore,
)

_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=UTC)
_LATER = datetime(2024, 6, 3, 15, 0, 0, tzinfo=UTC)


class TestInMemoryKillSwitchStore:
    @pytest.mark.asyncio
    async def test_initial_state_is_inactive(self) -> None:
        store = InMemoryKillSwitchStore()
        state = await store.get()
        assert state.active is False
        assert state.reason is None
        assert state.activated_at is None

    @pytest.mark.asyncio
    async def test_activate_sets_state(self) -> None:
        store = InMemoryKillSwitchStore()
        await store.activate(reason="test halt", activated_by="pytest", as_of=_NOW)
        state = await store.get()
        assert state.active is True
        assert state.reason == "test halt"
        assert state.activated_by == "pytest"
        assert state.activated_at == _NOW

    @pytest.mark.asyncio
    async def test_clear_resets_active_and_records_cleared_at(self) -> None:
        """clear() must record cleared_at for the kill-switch audit trail."""
        store = InMemoryKillSwitchStore()
        await store.activate(reason="halt", activated_by="auto", as_of=_NOW)
        await store.clear(operator_id="operator-1", as_of=_LATER)

        state = await store.get()
        assert state.active is False
        assert state.reason is None
        assert state.cleared_at == _LATER

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self) -> None:
        store = InMemoryKillSwitchStore()
        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_cleared_at_is_none_before_any_clear(self) -> None:
        store = InMemoryKillSwitchStore()
        state = await store.get()
        assert state.cleared_at is None

    @pytest.mark.asyncio
    async def test_activate_after_clear_reactivates(self) -> None:
        store = InMemoryKillSwitchStore()
        await store.activate(reason="first", activated_by="a", as_of=_NOW)
        await store.clear(operator_id="op", as_of=_NOW)
        await store.activate(reason="second", activated_by="b", as_of=_LATER)
        state = await store.get()
        assert state.active is True
        assert state.reason == "second"
