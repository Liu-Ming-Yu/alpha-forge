"""Shared strategy-engine loop orchestration.

This module keeps CLI adapters thin by owning the lifecycle that both
``run-engine`` and ``supervise`` need: initialize once, run ticks, isolate
per-cycle failures, sleep between supervised ticks, refresh durable
kill-switch state, and always shut the runner down.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.results import (
    ResultPresentation,
    UseCaseResult,
    UseCaseStatus,
)
from quant_platform.bootstrap.engine.loop_recovery import (
    assess_kill_switch_recovery,
    kill_switch_active,
    refresh_kill_switch_state,
)
from quant_platform.bootstrap.engine.loop_runtime import (
    close_event_bus,
    create_runner,
    engine_session,
    install_signal_handlers,
    load_and_validate_contracts,
    should_continue,
    should_sleep,
    sleep_until_next_cycle,
    validate_loop_config,
)
from quant_platform.bootstrap.engine.loop_types import (
    EngineLoopConfig,
    EngineLoopError,
    EngineLoopRunner,
    EngineLoopSummary,
    KillSwitchRecoveryAssessment,
    LoopProgress,
    RunnerFactory,
    SleepFn,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def run_engine_loop(
    settings: PlatformSettings,
    config: EngineLoopConfig,
    *,
    runner_factory: RunnerFactory | None = None,
    sleep_fn: SleepFn | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> EngineLoopSummary:
    """Run a strategy engine for a bounded count or until shutdown."""

    validate_loop_config(config)
    run_mode = config.mode
    backend = config.execution_backend
    contracts = load_and_validate_contracts(settings, config, run_mode, backend)
    from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema

    await verify_postgres_schema(settings)
    sleep = sleep_fn or sleep_until_next_cycle
    shutdown = shutdown_event or asyncio.Event()
    progress = LoopProgress()

    if runner_factory is None:

        def _default_runner() -> EngineLoopRunner:
            return create_runner(settings, config, run_mode, backend, contracts)

        runner_factory = _default_runner

    engine = runner_factory()
    restore_handlers = install_signal_handlers(shutdown) if config.install_signal_handlers else None
    try:
        await engine.initialize()
        while should_continue(progress.attempted_cycles, config.max_cycles, shutdown):
            session = engine_session(engine)
            if session is not None:
                await refresh_kill_switch_state(session)
            if session is not None and kill_switch_active(session):
                progress.attempted_cycles += 1
                progress.blocked_cycles += 1
                progress.last_recovery_assessment = await assess_kill_switch_recovery(session)
                log.warning(
                    "engine_loop.kill_switch_active",
                    engine=config.engine_name,
                    mode=config.mode,
                    ready_for_operator_clear=(
                        progress.last_recovery_assessment.ready_for_operator_clear
                    ),
                    detail=progress.last_recovery_assessment.detail,
                )
                if not should_sleep(progress.attempted_cycles, config.max_cycles, shutdown):
                    break
                await sleep(shutdown, config.interval_seconds)
                continue

            progress.attempted_cycles += 1
            try:
                result = await engine.run_cycle(feature_data={})
            except Exception as exc:
                progress.errors.append(
                    EngineLoopError(
                        cycle=progress.attempted_cycles,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                log.error(
                    "engine_loop.cycle_error",
                    engine=config.engine_name,
                    cycle=progress.attempted_cycles,
                    error=str(exc),
                    exc_info=True,
                )
            else:
                progress.completed_cycles += 1
                log.info(
                    "engine_loop.cycle_complete",
                    engine=config.engine_name,
                    cycle=progress.attempted_cycles,
                    signals=len(result.signals),
                    submitted=len(result.submitted_ids),
                    fills=len(result.fills),
                )

            if not should_sleep(progress.attempted_cycles, config.max_cycles, shutdown):
                break
            await sleep(shutdown, config.interval_seconds)

        progress.stop_reason = "shutdown_requested" if shutdown.is_set() else "max_cycles_reached"
    finally:
        if restore_handlers is not None:
            restore_handlers()
        with contextlib.suppress(Exception):
            await engine.shutdown()
        session = engine_session(engine)
        if session is not None:
            await close_event_bus(session)

    final_session = engine_session(engine)
    final_kill_switch_active = kill_switch_active(final_session) if final_session else False
    return EngineLoopSummary(
        engine_name=config.engine_name,
        mode=config.mode,
        execution_backend=config.execution_backend,
        attempted_cycles=progress.attempted_cycles,
        completed_cycles=progress.completed_cycles,
        blocked_cycles=progress.blocked_cycles,
        errors=tuple(progress.errors),
        stop_reason=progress.stop_reason,
        kill_switch_active=final_kill_switch_active,
        last_recovery_assessment=progress.last_recovery_assessment,
    )


def engine_loop_use_case_result(summary: EngineLoopSummary) -> UseCaseResult[dict[str, object]]:
    """Convert an engine-loop summary into a CLI/API use-case result."""

    status = UseCaseStatus.OK
    exit_code = 0
    message = "engine loop completed"
    if summary.errors:
        status = UseCaseStatus.FAILED
        exit_code = 1
        message = f"engine loop completed with {len(summary.errors)} cycle error(s)"
    elif summary.kill_switch_active and summary.completed_cycles == 0:
        status = UseCaseStatus.BLOCKED
        exit_code = 2
        message = "engine loop blocked by active kill switch"
    return UseCaseResult(
        status=status,
        payload=summary.as_payload(),
        message=message,
        exit_code=exit_code,
        presentation=ResultPresentation.JSON,
    )


__all__ = [
    "EngineLoopConfig",
    "EngineLoopError",
    "EngineLoopSummary",
    "KillSwitchRecoveryAssessment",
    "assess_kill_switch_recovery",
    "engine_loop_use_case_result",
    "refresh_kill_switch_state",
    "run_engine_loop",
]
