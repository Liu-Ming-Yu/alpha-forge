"""Broker session supervision and recovery."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.core.contracts import (
    BrokerHealthStatus,
    BrokerOrderRoutingGateway,
    BrokerSessionGateway,
    Clock,
    EventBus,
    ExecutionPolicy,
    OrderRepository,
)
from quant_platform.core.events import (
    BrokerSessionHealthChanged,
    KillSwitchActivated,
)
from quant_platform.services.execution_service.session.session_stale_orders import (
    cleanup_stale_orders,
)

if TYPE_CHECKING:
    from quant_platform.config import BrokerSettings


class ReconcileController(Protocol):
    async def reconcile(self, strategy_run_id: uuid.UUID) -> None: ...


log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SupervisorMetrics:
    broker_errors: int
    health_failures: int
    stale_orders_cancelled: int


class BrokerSessionSupervisor:
    """Supervise broker health and recovery for paper/live runtime sessions."""

    def __init__(
        self,
        session_gateway: BrokerSessionGateway,
        order_gateway: BrokerOrderRoutingGateway,
        event_bus: EventBus,
        execution_policy: ExecutionPolicy,
        reconcile_controller: ReconcileController,
        order_repo: OrderRepository,
        clock: Clock,
        settings: BrokerSettings,
    ) -> None:
        self._session_gateway = session_gateway
        self._order_gateway = order_gateway
        self._bus = event_bus
        self._policy = execution_policy
        self._reconcile_ctrl = reconcile_controller
        self._orders = order_repo
        self._clock = clock
        self._settings = settings

        self._started = False
        self._current_status = BrokerHealthStatus.DISCONNECTED
        self._health_failures = 0
        self._broker_errors = 0
        self._stale_orders_cancelled = 0
        self._last_health_poll_at: datetime | None = None

    @property
    def metrics(self) -> SupervisorMetrics:
        return SupervisorMetrics(
            broker_errors=self._broker_errors,
            health_failures=self._health_failures,
            stale_orders_cancelled=self._stale_orders_cancelled,
        )

    async def startup_recovery(self, strategy_run_id: uuid.UUID) -> bool:
        """Run startup sequence: connect, recover open orders, reconcile."""
        if self._started:
            return True
        ok = await self._recover_and_reconcile(strategy_run_id, phase="startup")
        self._started = ok
        if not ok:
            await self._activate_kill_switch(
                activated_by="session_supervisor",
                reason="startup recovery failed",
            )
        return ok

    async def poll_health(self, strategy_run_id: uuid.UUID) -> bool:
        """Check broker health and trigger reconnect on failures."""
        now = self._clock.now()
        if self._last_health_poll_at is not None:
            elapsed = (now - self._last_health_poll_at).total_seconds()
            if elapsed < self._settings.heartbeat_interval_seconds:
                return self._current_status == BrokerHealthStatus.CONNECTED
        self._last_health_poll_at = now

        try:
            health = await self._session_gateway.health_check()
        except Exception as exc:
            self._broker_errors += 1
            self._health_failures += 1
            await self._set_health_status(
                BrokerHealthStatus.DISCONNECTED,
                f"health_check failed: {exc}",
            )
            return await self._handle_failure(strategy_run_id, f"health check failed: {exc}")

        await self._set_health_status(health.status, health.detail)
        if health.status == BrokerHealthStatus.CONNECTED:
            self._health_failures = 0
            return True

        self._health_failures += 1
        return await self._handle_failure(
            strategy_run_id,
            f"broker health {health.status.value}: {health.detail}",
        )

    async def cleanup_stale_day_orders(self, strategy_run_id: uuid.UUID) -> int:
        """Cancel stale DAY orders; reservation release is lifecycle-driven."""
        result = await cleanup_stale_orders(
            session_gateway=self._session_gateway,
            order_gateway=self._order_gateway,
            order_repo=self._orders,
            now=self._clock.now(),
            strategy_run_id=strategy_run_id,
            day_threshold=timedelta(minutes=self._settings.stale_day_order_cleanup_minutes),
            gtc_threshold=(
                timedelta(minutes=self._settings.stale_gtc_cleanup_minutes)
                if self._settings.stale_gtc_cleanup_minutes > 0
                else None
            ),
        )
        self._broker_errors += result.broker_errors
        self._stale_orders_cancelled += result.cancelled
        return result.cancelled

    async def _handle_failure(self, strategy_run_id: uuid.UUID, reason: str) -> bool:
        recovered = await self._recover_and_reconcile(strategy_run_id, phase="reconnect")
        if recovered:
            self._health_failures = 0
            return True

        if self._health_failures >= self._settings.max_consecutive_health_failures:
            await self._activate_kill_switch(
                activated_by="session_supervisor",
                reason=(
                    f"max broker health failures reached: {self._health_failures}; last={reason}"
                ),
            )
        return False

    async def _recover_and_reconcile(self, strategy_run_id: uuid.UUID, phase: str) -> bool:
        delay = self._settings.reconnect_base_delay
        attempts = max(1, self._settings.max_consecutive_health_failures)

        for attempt in range(1, attempts + 1):
            try:
                await self._session_gateway.connect()
                connect_order_gateway = getattr(self._order_gateway, "connect", None)
                if id(self._order_gateway) != id(self._session_gateway) and connect_order_gateway:
                    await connect_order_gateway()

                await self._session_gateway.fetch_open_orders()

                if self._settings.reconcile_on_reconnect and hasattr(
                    self._reconcile_ctrl, "reconcile"
                ):
                    try:
                        await self._reconcile_ctrl.reconcile(strategy_run_id)
                    except Exception as exc:
                        await self._activate_kill_switch(
                            activated_by="session_supervisor",
                            reason=f"reconcile after reconnect failed: {exc}",
                        )
                        await self._set_health_status(
                            BrokerHealthStatus.DEGRADED,
                            "reconcile_after_reconnect failed",
                        )
                        return False
                    if self._policy.kill_switch_active:
                        await self._set_health_status(
                            BrokerHealthStatus.DEGRADED,
                            "reconcile_after_reconnect requested operator action",
                        )
                        return False

                await self._set_health_status(
                    BrokerHealthStatus.CONNECTED,
                    f"{phase} recovery successful",
                )
                return True
            except Exception as exc:
                self._broker_errors += 1
                await self._set_health_status(
                    BrokerHealthStatus.DISCONNECTED,
                    f"{phase} attempt {attempt} failed: {exc}",
                )
                log.warning(
                    "session_supervisor.recovery_failed",
                    phase=phase,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._settings.reconnect_max_delay)

        return False

    async def _set_health_status(
        self,
        status: BrokerHealthStatus,
        detail: str = "",
    ) -> None:
        if status == self._current_status:
            return
        previous = self._current_status
        self._current_status = status
        await self._bus.publish(
            BrokerSessionHealthChanged(
                event_id=uuid.uuid4(),
                occurred_at=self._clock.now(),
                previous_status=previous.value,
                current_status=status.value,
                detail=detail,
            )
        )

    async def _activate_kill_switch(self, activated_by: str, reason: str) -> None:
        if hasattr(self._policy, "activate_kill_switch_durable"):
            await self._policy.activate_kill_switch_durable(reason, activated_by=activated_by)
        elif hasattr(self._policy, "activate_kill_switch"):
            self._policy.activate_kill_switch(reason, activated_by=activated_by)
        await self._bus.publish(
            KillSwitchActivated(
                event_id=uuid.uuid4(),
                occurred_at=self._clock.now(),
                activated_by=activated_by,
                reason=reason,
            )
        )
