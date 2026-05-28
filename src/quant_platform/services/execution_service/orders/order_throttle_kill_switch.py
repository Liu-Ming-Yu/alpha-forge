"""Kill-switch behavior for the order throttle."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_platform.core.contracts import Clock
    from quant_platform.services.execution_service.stores.kill_switch_store import KillSwitchStore


class OrderThrottleKillSwitchMixin:
    """Durable kill-switch operations shared by order throttles."""

    _clock: Clock
    _kill_switch_active: bool
    _kill_switch_reason: str
    _kill_switch_store: KillSwitchStore | None

    def activate_kill_switch(self, reason: str, *, activated_by: str) -> None:
        """Activate the kill switch, blocking all further submissions."""
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._schedule_store_write(activate=True, reason=reason, activated_by=activated_by)

    async def activate_kill_switch_durable(self, reason: str, *, activated_by: str) -> None:
        """Activate the kill switch with store-first durability guarantee."""
        store = self._kill_switch_store
        if store is not None:
            await store.activate(
                reason=reason,
                activated_by=activated_by,
                as_of=self._clock.now(),
            )
        self._kill_switch_active = True
        self._kill_switch_reason = reason

    def clear_kill_switch(self, operator_id: str) -> None:
        """Clear the kill switch to allow submissions to resume."""
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._schedule_store_write(activate=False, reason=None, activated_by=operator_id)

    def hydrate_kill_switch(
        self,
        *,
        active: bool | None = None,
        reason: str | None = None,
    ) -> object | None:
        """Restore kill-switch state from the durable store at startup."""
        if active is not None:
            self._kill_switch_active = bool(active)
            self._kill_switch_reason = reason or ""
            return None

        async def _hydrate_from_store() -> None:
            store = self._kill_switch_store
            if store is None:
                self._kill_switch_active = False
                self._kill_switch_reason = ""
                return
            state = await store.get()
            self._kill_switch_active = bool(state.active) if state is not None else False
            self._kill_switch_reason = (state.reason if state is not None else None) or ""

        return _hydrate_from_store()

    def _schedule_store_write(
        self,
        *,
        activate: bool,
        reason: str | None,
        activated_by: str,
    ) -> None:
        """Best-effort persistence of the kill-switch state."""
        store = self._kill_switch_store
        if store is None:
            return
        try:
            import asyncio

            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if activate:
            coro = store.activate(
                reason=reason or "",
                activated_by=activated_by,
                as_of=self._clock.now(),
            )
        else:
            coro = store.clear(
                operator_id=activated_by,
                as_of=self._clock.now(),
            )
        loop.create_task(coro)
