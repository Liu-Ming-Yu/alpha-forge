"""Research operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_platform.application.results import UseCaseResult
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import (
        FactorsCalibrateRequest,
        TearsheetRequest,
    )
    from quant_platform.application.research import (
        AlphaRequest,
        BacktestRequest,
        BoostingRequest,
        CampaignRequest,
        FeaturesRequest,
        FeaturesRetentionRequest,
        ModelRegistryRequest,
        WalkForwardRequest,
    )


class ResearchUseCasePorts(Protocol):
    """Research adapters required by operator use cases."""

    def factors_calibrate(self, request: FactorsCalibrateRequest) -> Awaitable[None]: ...

    def tearsheet(self, request: TearsheetRequest) -> Awaitable[None]: ...

    def model_registry(self, request: ModelRegistryRequest) -> Awaitable[None]: ...

    def boosting(self, request: BoostingRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def alpha(self, request: AlphaRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def walk_forward(
        self, request: WalkForwardRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def features(self, request: FeaturesRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def features_retention(
        self, request: FeaturesRetentionRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def campaign(self, request: CampaignRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def backtest(self, request: BacktestRequest) -> Awaitable[UseCaseResult[dict[str, object]]]: ...


def register_research_use_cases(registry: UseCaseRegistry, ports: ResearchUseCasePorts) -> None:
    """Register research use cases."""

    async def factors(request: FactorsCalibrateRequest) -> UseCaseResult[None]:
        await ports.factors_calibrate(request)
        return UseCaseResult()

    async def tearsheet(request: TearsheetRequest) -> UseCaseResult[None]:
        await ports.tearsheet(request)
        return UseCaseResult()

    async def model_registry(request: ModelRegistryRequest) -> UseCaseResult[None]:
        await ports.model_registry(request)
        return UseCaseResult()

    async def boosting(request: BoostingRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.boosting(request)

    async def alpha(request: AlphaRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.alpha(request)

    async def walk_forward(request: WalkForwardRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.walk_forward(request)

    async def features(request: FeaturesRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.features(request)

    async def features_retention(
        request: FeaturesRetentionRequest,
    ) -> UseCaseResult[dict[str, object]]:
        return await ports.features_retention(request)

    async def campaign(request: CampaignRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.campaign(request)

    async def backtest(request: BacktestRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.backtest(request)

    registry.register("research.factors_calibrate", CallableUseCase(factors))
    registry.register("research.tearsheet", CallableUseCase(tearsheet))
    registry.register("research.model_registry", CallableUseCase(model_registry))
    registry.register("research.boosting", CallableUseCase(boosting))
    registry.register("research.alpha", CallableUseCase(alpha))
    registry.register("research.walk_forward", CallableUseCase(walk_forward))
    registry.register("research.features", CallableUseCase(features))
    registry.register("research.features_retention", CallableUseCase(features_retention))
    registry.register("research.campaign", CallableUseCase(campaign))
    registry.register("research.backtest", CallableUseCase(backtest))


__all__ = ["ResearchUseCasePorts", "register_research_use_cases"]
