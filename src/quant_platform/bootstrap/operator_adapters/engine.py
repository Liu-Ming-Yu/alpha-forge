"""Strategy-engine operator adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.application.operator.cli_inputs import load_instrument_contracts
from quant_platform.application.results import UseCaseResult
from quant_platform.bootstrap.engine.loop import (
    EngineLoopConfig,
    engine_loop_use_case_result,
    run_engine_loop,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.application.operator.requests import RunEngineRequest, RunMultiEngineRequest
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


class EngineAdapters:
    """Concrete strategy-engine adapters backed by engine bootstrap helpers."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def run_engine(self, request: RunEngineRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.bootstrap.engine import run_multi_engine_v2
        from quant_platform.engines.engine_runner import ExecutionBackend, RunMode

        run_mode = RunMode(request.mode)
        backend = ExecutionBackend(request.execution_backend)
        if backend == ExecutionBackend.IB_PAPER and run_mode != RunMode.PAPER:
            raise ValueError("--execution-backend ib-paper is only valid with --mode paper")
        if backend == ExecutionBackend.IB_PAPER and not request.contracts_file:
            raise ValueError("--execution-backend ib-paper requires --contracts-file")
        if (
            self._settings.v2.enabled
            and self._settings.v2.account_orchestrator_enabled
            and run_mode == RunMode.LIVE
        ):
            await run_multi_engine_v2(
                self._settings,
                mode=request.mode,
                engine_names=[request.engine_name],
                budgets_file=None,
                cycles=request.cycles,
                initial_cash=request.initial_cash,
                instrument_contracts=load_instrument_contracts(request.contracts_file)
                if request.contracts_file
                else {},
            )
            return UseCaseResult()

        summary = await run_engine_loop(
            self._settings,
            EngineLoopConfig(
                engine_name=request.engine_name,
                mode=request.mode,
                execution_backend=backend.value,
                initial_cash=request.initial_cash,
                contracts_file=request.contracts_file,
                interval_seconds=0.0,
                max_cycles=request.cycles,
                install_signal_handlers=False,
            ),
        )
        log.info("engine.complete", **summary.as_payload())
        return engine_loop_use_case_result(summary)

    async def run_multi_engine(self, request: RunMultiEngineRequest) -> None:
        from quant_platform.bootstrap.engine import run_multi_engine_v2

        contracts: dict[uuid.UUID, dict[str, object]] = {}
        if (
            self._settings.v2.enabled
            and self._settings.v2.account_orchestrator_enabled
            and request.contracts_file
        ):
            contracts = load_instrument_contracts(request.contracts_file)
        await run_multi_engine_v2(
            self._settings,
            mode=request.mode,
            engine_names=list(request.engine_names),
            budgets_file=request.budgets_file,
            cycles=request.cycles,
            initial_cash=request.initial_cash,
            instrument_contracts=contracts,
        )


__all__ = ["EngineAdapters"]
