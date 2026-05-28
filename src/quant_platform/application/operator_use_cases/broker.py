"""Broker operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_platform.application.results import (
    ResultPresentation,
    UseCaseResult,
    UseCaseStatus,
)
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import (
        BrokerContractsRequest,
        EventBusSweepRequest,
        PaperLifecycleRequest,
        PassiveRepriceRequest,
    )


class BrokerUseCasePorts(Protocol):
    """Broker adapters required by operator use cases."""

    def gateway_smoke(self, request: BrokerContractsRequest) -> Awaitable[dict[str, object]]: ...

    def paper_lifecycle(self, request: PaperLifecycleRequest) -> Awaitable[dict[str, object]]: ...

    def passive_reprice(self, request: PassiveRepriceRequest) -> Awaitable[dict[str, object]]: ...

    def sweep_dead_letters(self, request: EventBusSweepRequest) -> Awaitable[tuple[int, int]]: ...


def register_broker_use_cases(registry: UseCaseRegistry, ports: BrokerUseCasePorts) -> None:
    """Register broker use cases."""

    async def gateway_smoke(request: BrokerContractsRequest) -> UseCaseResult[dict[str, object]]:
        return _json_assertion_result(await ports.gateway_smoke(request))

    async def paper_lifecycle(request: PaperLifecycleRequest) -> UseCaseResult[dict[str, object]]:
        return _json_assertion_result(await ports.paper_lifecycle(request))

    async def passive_reprice(request: PassiveRepriceRequest) -> UseCaseResult[dict[str, object]]:
        return UseCaseResult(
            payload=await ports.passive_reprice(request),
            presentation=ResultPresentation.JSON,
        )

    async def sweep(request: EventBusSweepRequest) -> UseCaseResult[str]:
        moved, depth = await ports.sweep_dead_letters(request)
        return UseCaseResult(
            message=f"moved={moved} dlq_stream={request.stream}.dlq depth={depth}",
            presentation=ResultPresentation.TEXT,
        )

    registry.register("broker.ib_gateway_smoke", CallableUseCase(gateway_smoke))
    registry.register("broker.ib_paper_lifecycle", CallableUseCase(paper_lifecycle))
    registry.register("broker.passive_reprice_once", CallableUseCase(passive_reprice))
    registry.register("event_bus.sweep_dead_letters", CallableUseCase(sweep))


def _json_assertion_result(report: dict[str, object]) -> UseCaseResult[dict[str, object]]:
    passed = bool(report.get("passed"))
    return UseCaseResult(
        status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
        payload=report,
        exit_code=0 if passed else 2,
        presentation=ResultPresentation.JSON,
    )


__all__ = ["BrokerUseCasePorts", "register_broker_use_cases"]
