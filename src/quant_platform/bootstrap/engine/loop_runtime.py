"""Runtime helpers for bounded and supervised engine loops."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog

from quant_platform.application.operator.cli_inputs import load_instrument_contracts
from quant_platform.bootstrap.broker.live_broker_wiring import validate_ib_paper_execution

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from quant_platform.application.runtime.state import CycleResult, Session
    from quant_platform.bootstrap.engine.loop_types import EngineLoopConfig, EngineLoopRunner
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


class _FactoryInitializedRunner(Protocol):
    """Concrete engine shape before adapting to the loop runner protocol."""

    async def initialize(self, *, session_factory: object, scheduler_factory: object) -> None: ...

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
    ) -> CycleResult: ...

    async def shutdown(self) -> object: ...


def validate_loop_config(config: EngineLoopConfig) -> None:
    """Validate command-level loop configuration before any side effects."""

    if config.mode not in {"shadow", "paper", "live"}:
        raise ValueError(f"invalid --mode {config.mode!r}")
    if config.execution_backend not in {"simulated", "ib-paper"}:
        raise ValueError(f"invalid --execution-backend {config.execution_backend!r}")
    if config.interval_seconds < 0:
        raise ValueError("--interval must be >= 0")
    if config.max_cycles is not None and config.max_cycles <= 0:
        raise ValueError("--cycles/--max-cycles must be > 0")


def load_and_validate_contracts(
    settings: PlatformSettings,
    config: EngineLoopConfig,
    run_mode: str,
    backend: str,
) -> dict[uuid.UUID, dict[str, object]]:
    """Load optional contracts and enforce broker/mode constraints."""

    if backend == "ib-paper" and run_mode != "paper":
        raise ValueError("--execution-backend ib-paper is only valid with --mode paper")
    if backend == "ib-paper" and not config.contracts_file:
        raise ValueError("--execution-backend ib-paper requires --contracts-file")
    if run_mode == "live" and not config.contracts_file:
        raise ValueError(
            "LIVE mode requires --contracts-file pointing to JSON "
            "mapping instrument_id UUID -> contract spec"
        )
    contracts = load_instrument_contracts(config.contracts_file) if config.contracts_file else {}
    if backend == "ib-paper":
        validate_ib_paper_execution(settings, contracts)
    return contracts


def create_runner(
    settings: PlatformSettings,
    config: EngineLoopConfig,
    run_mode: str,
    backend: str,
    contracts: dict[uuid.UUID, dict[str, object]],
) -> EngineLoopRunner:
    """Build the default runner after config and schema checks pass."""

    from quant_platform.bootstrap.engine.multi import create_single_engine_runner

    runner = cast(
        "_FactoryInitializedRunner",
        create_single_engine_runner(
            settings=settings,
            engine_name=config.engine_name,
            mode=run_mode,
            execution_backend=backend,
            initial_cash=config.initial_cash,
            instrument_contracts=contracts,
        ),
    )
    return _DefaultEngineLoopRunner(runner)


class _DefaultEngineLoopRunner:
    """Adapt ``EngineRunner`` to the zero-argument loop runner protocol."""

    def __init__(self, runner: _FactoryInitializedRunner) -> None:
        self._runner = runner

    @property
    def _session(self) -> Session | None:
        return getattr(self._runner, "_session", None)

    async def initialize(self) -> None:
        from quant_platform.bootstrap.engine.session_wiring import (
            build_engine_maintenance_scheduler,
            create_engine_runtime_session,
        )

        await self._runner.initialize(
            session_factory=create_engine_runtime_session,
            scheduler_factory=build_engine_maintenance_scheduler,
        )

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
    ) -> CycleResult:
        return await self._runner.run_cycle(feature_data=feature_data)

    async def shutdown(self) -> object:
        return await self._runner.shutdown()


def should_continue(
    attempted_cycles: int,
    max_cycles: int | None,
    shutdown: asyncio.Event,
) -> bool:
    """Return whether another cycle should start."""

    if shutdown.is_set():
        return False
    return max_cycles is None or attempted_cycles < max_cycles


def should_sleep(attempted_cycles: int, max_cycles: int | None, shutdown: asyncio.Event) -> bool:
    """Return whether the loop should wait before checking another cycle."""

    if shutdown.is_set():
        return False
    return max_cycles is None or attempted_cycles < max_cycles


async def sleep_until_next_cycle(shutdown: asyncio.Event, interval_seconds: float) -> None:
    """Sleep for the interval or wake early when shutdown is requested."""

    if interval_seconds <= 0:
        return
    sleep_task = asyncio.create_task(asyncio.sleep(interval_seconds))
    shutdown_task = asyncio.create_task(shutdown.wait())
    _done, pending = await asyncio.wait(
        {sleep_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def install_signal_handlers(shutdown: asyncio.Event) -> Callable[[], None]:
    """Install temporary SIGTERM/SIGINT handlers and return a restore callback."""

    previous: dict[int, Any] = {}

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("engine_loop.shutdown_requested", signal=signum)
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(shutdown.set)
        except RuntimeError:
            shutdown.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, _on_signal)
        except (ValueError, OSError):
            continue

    def _restore() -> None:
        for signum, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signum, handler)

    return _restore


def engine_session(engine: object) -> Session | None:
    """Return the runner's session when it exposes the historical private slot."""

    session = getattr(engine, "_session", None)
    return session if session is not None else None


async def close_event_bus(session: Session) -> None:
    """Close an async event bus if the session owns one."""

    bus_aclose = getattr(session.event_bus, "aclose", None)
    if bus_aclose is None:
        return
    try:
        await bus_aclose()
    except Exception as exc:  # pragma: no cover - protective shutdown path
        log.warning("engine_loop.event_bus_close_failed", error=str(exc))


__all__ = [
    "close_event_bus",
    "create_runner",
    "engine_session",
    "install_signal_handlers",
    "load_and_validate_contracts",
    "should_continue",
    "should_sleep",
    "sleep_until_next_cycle",
    "validate_loop_config",
]
