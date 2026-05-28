"""Research operator adapters.

Each method translates a typed research request DTO into an explicit research
composition operation. The typed request is the contract end to end: there is
no ``SimpleNamespace`` bridge. Workflows that emit an operator-facing JSON
payload return a ``UseCaseResult`` so the CLI presentation layer owns
print/exit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


class ResearchAdapters:
    """Concrete research adapters backed by research composition operations."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def factors_calibrate(self, request: FactorsCalibrateRequest) -> None:
        from quant_platform.research import ops as research_ops

        await research_ops._factors_calibrate(
            self._settings,
            request.samples_path,
            request.output_dir,
            request.horizon_days,
            request.l2_lambda,
            request.momentum_scale,
        )

    async def tearsheet(self, request: TearsheetRequest) -> None:
        from quant_platform.research import ops as research_ops

        await research_ops._tearsheet(self._settings, request.run_id, request.root)

    async def model_registry(self, request: ModelRegistryRequest) -> None:
        from quant_platform.research import ops as research_ops

        await research_ops._model_registry(self._settings, request)

    async def boosting(self, request: BoostingRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._boosting(self._settings, request)

    async def alpha(self, request: AlphaRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.bootstrap.governance import alpha_command

        return await alpha_command(self._settings, request)

    async def walk_forward(self, request: WalkForwardRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._walk_forward(self._settings, request)

    async def features(self, request: FeaturesRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._features(self._settings, request)

    async def features_retention(
        self, request: FeaturesRetentionRequest
    ) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._features_retention(
            self._settings, request.keep_days, request.dry_run
        )

    async def campaign(self, request: CampaignRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._research_campaign(self._settings, request)

    async def backtest(self, request: BacktestRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.research import ops as research_ops

        return await research_ops._backtest_run(self._settings, request)


__all__ = ["ResearchAdapters"]
