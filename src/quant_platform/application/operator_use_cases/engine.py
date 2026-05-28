"""Strategy-engine operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_platform.application.results import UseCaseResult
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import RunEngineRequest, RunMultiEngineRequest


class EngineUseCasePorts(Protocol):
    """Engine adapters required by operator use cases."""

    def run_engine(self, request: RunEngineRequest) -> Awaitable[UseCaseResult[object] | None]: ...

    def run_multi_engine(
        self,
        request: RunMultiEngineRequest,
    ) -> Awaitable[UseCaseResult[object] | None]: ...


def register_engine_use_cases(registry: UseCaseRegistry, ports: EngineUseCasePorts) -> None:
    """Register strategy-engine use cases."""

    async def run_engine(request: RunEngineRequest) -> UseCaseResult[object]:
        result = await ports.run_engine(request)
        if isinstance(result, UseCaseResult):
            return result
        return UseCaseResult()

    async def run_multi_engine(request: RunMultiEngineRequest) -> UseCaseResult[object]:
        result = await ports.run_multi_engine(request)
        if isinstance(result, UseCaseResult):
            return result
        return UseCaseResult()

    registry.register("engine.run", CallableUseCase(run_engine))
    registry.register("engine.run_multi", CallableUseCase(run_multi_engine))


__all__ = ["EngineUseCasePorts", "register_engine_use_cases"]
