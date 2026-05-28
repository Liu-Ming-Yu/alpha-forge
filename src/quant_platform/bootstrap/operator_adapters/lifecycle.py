"""Lifecycle operator adapters: run-cycle, supervise, serve-api, smoke."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus
from quant_platform.bootstrap.engine.loop import (
    engine_loop_use_case_result,
)

if TYPE_CHECKING:
    from quant_platform.application.operator.requests import (
        NoInputRequest,
        RunCycleRequest,
        ServeApiRequest,
        SuperviseRequest,
    )
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


class RuntimeAdapters:
    """Concrete runtime adapters backed by bootstrap composition functions."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def run_cycle(self, request: RunCycleRequest) -> None:
        from quant_platform.bootstrap.engine import run_cycle_once

        result = await run_cycle_once(self._settings, initial_cash=request.initial_cash)
        log.info(
            "run_cycle.result",
            signals=len(result.signals),
            approved=len(result.approved),
            rejected=len(result.rejected),
            submitted=len(result.submitted_ids),
            fills=len(result.fills),
        )

    async def supervise(self, request: SuperviseRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.bootstrap.engine import supervise_engine

        summary = await supervise_engine(
            self._settings,
            initial_cash=request.initial_cash,
            interval_seconds=request.interval_seconds,
            mode=request.mode,
            max_cycles=request.max_cycles,
            contracts_file=request.contracts_file,
            engine_name=request.engine_name,
            execution_backend=request.execution_backend,
        )
        return engine_loop_use_case_result(summary)

    async def health(self, _request: NoInputRequest) -> dict[str, object]:
        from quant_platform.bootstrap.broker import broker_health

        return await broker_health(self._settings)

    async def serve_api(self, request: ServeApiRequest) -> UseCaseResult[str]:
        try:
            import uvicorn
        except ImportError:
            return UseCaseResult(
                status=UseCaseStatus.FAILED,
                message="uvicorn is required: pip install -e '.[api]'",
                exit_code=1,
                presentation=ResultPresentation.TEXT,
            )

        from quant_platform.bootstrap.operator_api.app import build_operator_api_app
        from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema

        await verify_postgres_schema(self._settings)
        app = build_operator_api_app(self._settings, initial_cash=request.initial_cash)
        config = uvicorn.Config(app, host=request.host, port=request.port)
        server = uvicorn.Server(config)
        await server.serve()
        return UseCaseResult()

    async def smoke(self, _request: NoInputRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.bootstrap.governance.commands import smoke_command

        return await smoke_command(self._settings)


__all__ = ["RuntimeAdapters"]
