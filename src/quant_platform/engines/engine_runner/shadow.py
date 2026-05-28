"""Shadow-cycle mixin for ``EngineRunner``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.engines.shadow.boosting_cycle import (
    run_shadow_boosting_cycle as run_shadow_boosting_scoring_cycle,
)
from quant_platform.engines.shadow.cycle import run_shadow_target_cycle
from quant_platform.engines.shadow.text_cycle import (
    run_shadow_text_cycle as run_shadow_text_scoring_cycle,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.application.runtime.state import CycleResult, Session
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.core.domain.signals import SignalScore
    from quant_platform.engines.framework.types import EngineConfig, EngineRunResult
    from quant_platform.engines.shadow.boosting_cycle import ShadowBoostingScorer
    from quant_platform.engines.shadow.text_cycle import ShadowTextCycleScorer


class EngineRunnerShadowMixin:
    """Shadow target and auxiliary scoring adapters."""

    _config: EngineConfig
    _session: Session | None
    _strategy_run: StrategyRun | None
    _result: EngineRunResult
    _shadow_boosting_scorer: ShadowBoostingScorer | None
    _shadow_text_scorer: ShadowTextCycleScorer | None

    async def _run_shadow_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
        market_prices: dict[uuid.UUID, Decimal] | None = None,
    ) -> CycleResult:
        session = self._session
        strategy_run = self._strategy_run
        if session is None or strategy_run is None:
            raise RuntimeError("call initialize() first")

        result = await run_shadow_target_cycle(
            session=session,
            strategy_run=strategy_run,
            feature_data=feature_data,
            market_prices=market_prices,
            engine_name=self._config.engine_name,
        )
        self._result.cycles_completed += 1
        self._result.total_signals += len(result.signals)
        self._result.shadow_only = True
        return result

    async def _run_shadow_boosting_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
        primary_scores: Sequence[SignalScore],
        as_of: datetime,
    ) -> None:
        await run_shadow_boosting_scoring_cycle(
            scorer=self._shadow_boosting_scorer,
            strategy_run=self._strategy_run,
            feature_data=feature_data,
            primary_scores=primary_scores,
            as_of=as_of,
            engine_name=self._config.engine_name,
        )

    async def _run_shadow_text_cycle(
        self,
        as_of: datetime,
        market_prices: dict[uuid.UUID, Decimal] | None = None,
    ) -> None:
        session = self._session
        if session is None:
            raise RuntimeError("call initialize() first")
        await run_shadow_text_scoring_cycle(
            scorer=self._shadow_text_scorer,
            session=session,
            strategy_run=self._strategy_run,
            as_of=as_of,
            market_prices=market_prices,
        )
