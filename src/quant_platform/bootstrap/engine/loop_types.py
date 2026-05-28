"""Shared types for strategy-engine loop orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import uuid
    from decimal import Decimal

    from quant_platform.application.runtime.state import CycleResult


class EngineLoopRunner(Protocol):
    """Subset of ``EngineRunner`` used by the loop."""

    async def initialize(self) -> None: ...

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
    ) -> CycleResult: ...

    async def shutdown(self) -> object: ...


RunnerFactory = Callable[[], EngineLoopRunner]
SleepFn = Callable[[asyncio.Event, float], Awaitable[None]]


@dataclass(frozen=True)
class EngineLoopConfig:
    """Configuration for bounded or continuous engine supervision."""

    engine_name: str
    mode: str
    execution_backend: str
    initial_cash: Decimal
    contracts_file: str | None
    interval_seconds: float = 0.0
    max_cycles: int | None = 1
    install_signal_handlers: bool = True


@dataclass(frozen=True)
class EngineLoopError:
    """One isolated cycle failure."""

    cycle: int
    error_type: str
    message: str

    def as_payload(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class KillSwitchRecoveryAssessment:
    """Read-only readiness check while the kill switch is active."""

    active: bool
    ready_for_operator_clear: bool
    broker_connected: bool
    open_orders: int | None
    operator_discrepancies: int | None
    cash_drift_ok: bool | None
    detail: str

    def as_payload(self) -> dict[str, object]:
        return {
            "active": self.active,
            "ready_for_operator_clear": self.ready_for_operator_clear,
            "broker_connected": self.broker_connected,
            "open_orders": self.open_orders,
            "operator_discrepancies": self.operator_discrepancies,
            "cash_drift_ok": self.cash_drift_ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EngineLoopSummary:
    """Operator-facing summary for a bounded or supervised engine loop."""

    engine_name: str
    mode: str
    execution_backend: str
    attempted_cycles: int = 0
    completed_cycles: int = 0
    blocked_cycles: int = 0
    errors: tuple[EngineLoopError, ...] = ()
    stop_reason: str = "unknown"
    kill_switch_active: bool = False
    last_recovery_assessment: KillSwitchRecoveryAssessment | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "engine": self.engine_name,
            "mode": self.mode,
            "execution_backend": self.execution_backend,
            "attempted_cycles": self.attempted_cycles,
            "completed_cycles": self.completed_cycles,
            "blocked_cycles": self.blocked_cycles,
            "errors": [error.as_payload() for error in self.errors],
            "stop_reason": self.stop_reason,
            "kill_switch_active": self.kill_switch_active,
            "last_recovery_assessment": (
                self.last_recovery_assessment.as_payload()
                if self.last_recovery_assessment is not None
                else None
            ),
        }


@dataclass
class LoopProgress:
    """Mutable counters accumulated by one engine loop run."""

    attempted_cycles: int = 0
    completed_cycles: int = 0
    blocked_cycles: int = 0
    errors: list[EngineLoopError] = field(default_factory=list)
    stop_reason: str = "unknown"
    last_recovery_assessment: KillSwitchRecoveryAssessment | None = None


__all__ = [
    "EngineLoopConfig",
    "EngineLoopError",
    "EngineLoopRunner",
    "EngineLoopSummary",
    "KillSwitchRecoveryAssessment",
    "LoopProgress",
    "RunnerFactory",
    "SleepFn",
]
