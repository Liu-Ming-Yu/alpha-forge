"""Runtime operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import (
        NoInputRequest,
        RunCycleRequest,
        ServeApiRequest,
        SuperviseRequest,
    )


class RuntimeUseCasePorts(Protocol):
    """Runtime adapters required by operator use cases."""

    def run_cycle(self, request: RunCycleRequest) -> Awaitable[None]: ...

    def supervise(self, request: SuperviseRequest) -> Awaitable[UseCaseResult[object] | None]: ...

    def health(self, request: NoInputRequest) -> Awaitable[dict[str, object]]: ...

    def serve_api(self, request: ServeApiRequest) -> Awaitable[UseCaseResult[str]]: ...

    def smoke(self, request: NoInputRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...


def register_runtime_use_cases(registry: UseCaseRegistry, ports: RuntimeUseCasePorts) -> None:
    """Register runtime use cases."""

    async def run_cycle(request: RunCycleRequest) -> UseCaseResult[None]:
        await ports.run_cycle(request)
        return UseCaseResult()

    async def supervise(request: SuperviseRequest) -> UseCaseResult[object]:
        result = await ports.supervise(request)
        if isinstance(result, UseCaseResult):
            return result
        return UseCaseResult()

    async def health(request: NoInputRequest) -> UseCaseResult[dict[str, object]]:
        return UseCaseResult(
            payload=await ports.health(request),
            presentation=ResultPresentation.KEY_VALUE,
        )

    async def smoke(request: NoInputRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.smoke(request)

    registry.register("runtime.run_cycle", CallableUseCase(run_cycle))
    registry.register("runtime.supervise", CallableUseCase(supervise))
    registry.register("runtime.health", CallableUseCase(health))
    registry.register("runtime.serve_api", CallableUseCase(ports.serve_api))
    registry.register("runtime.smoke", CallableUseCase(smoke))


__all__ = ["RuntimeUseCasePorts", "register_runtime_use_cases"]
