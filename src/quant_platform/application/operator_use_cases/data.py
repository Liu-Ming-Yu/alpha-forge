"""Data operator use cases."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from quant_platform.application.data import (
    ComputeFeaturesRequest,
    ComputeFeaturesUseCase,
    DataHealthRequest,
    DataHealthUseCase,
    IngestRequest,
    MaintainDataRequest,
    MaintainDataUseCase,
    ReprocessCorporateActionsRequest,
    ReprocessCorporateActionsUseCase,
)
from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import IntradayCommandRequest


class DataUseCasePorts(Protocol):
    """Data adapters required by operator use cases."""

    def compute_features(self, request: ComputeFeaturesRequest) -> Awaitable[None]: ...

    def ingest(self, request: IngestRequest) -> Awaitable[UseCaseResult[None]]: ...

    def maintain(self, request: MaintainDataRequest) -> Awaitable[None]: ...

    def reprocess_ca(self, request: ReprocessCorporateActionsRequest) -> Awaitable[None]: ...

    def intraday(
        self, request: IntradayCommandRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def health(self, request: DataHealthRequest) -> Awaitable[tuple[dict[str, object], bool]]: ...


def register_data_use_cases(registry: UseCaseRegistry, ports: DataUseCasePorts) -> None:
    """Register data use cases."""

    async def compute(request: ComputeFeaturesRequest) -> UseCaseResult[dict[str, object]]:
        return await ComputeFeaturesUseCase(runner=ports.compute_features).run(request)

    async def ingest(request: IngestRequest) -> UseCaseResult[None]:
        return await ports.ingest(request)

    async def maintain(request: MaintainDataRequest) -> UseCaseResult[dict[str, object]]:
        return await MaintainDataUseCase(runner=ports.maintain).run(request)

    async def reprocess(
        request: ReprocessCorporateActionsRequest,
    ) -> UseCaseResult[dict[str, object]]:
        return await ReprocessCorporateActionsUseCase(runner=ports.reprocess_ca).run(request)

    async def intraday(
        request: IntradayCommandRequest,
    ) -> UseCaseResult[dict[str, object]]:
        return await ports.intraday(request)

    async def data_health(request: DataHealthRequest) -> UseCaseResult[dict[str, object]]:
        result = await DataHealthUseCase(reporter=ports.health).run(request)
        return replace(result, presentation=ResultPresentation.JSON)

    registry.register("data.compute_features", CallableUseCase(compute))
    registry.register("data.ingest", CallableUseCase(ingest))
    registry.register("data.maintain", CallableUseCase(maintain))
    registry.register("data.reprocess_ca", CallableUseCase(reprocess))
    registry.register("data.intraday", CallableUseCase(intraday))
    registry.register("data.health", CallableUseCase(data_health))


__all__ = ["DataUseCasePorts", "register_data_use_cases"]
