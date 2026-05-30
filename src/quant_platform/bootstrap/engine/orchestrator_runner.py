"""Account-orchestrator loop runner (unified single/multi-engine execution).

This adapts the V2 account orchestrator to the ``EngineLoopRunner`` protocol so
the *same* robust loop (``run_engine_loop`` — kill-switch refresh, recovery
assessment, interval sleep, signal handling, per-cycle error isolation) drives
both ``run-multi-engine`` and ``supervise``. A single engine is just the N=1
case: one engine, one budget, one set of governance/read-model writes.

See ADR-014 (unified engine runtime).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.bootstrap.engine.session_wiring import (
    build_engine_maintenance_scheduler,
    create_engine_runtime_session,
)
from quant_platform.core.domain.research import RunStatus
from quant_platform.engines.market_data.price_seeding import latest_contract_market_prices

if TYPE_CHECKING:
    from decimal import Decimal

    from quant_platform.application.runtime.state import CycleResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import EngineBudget

log = structlog.get_logger(__name__)


class AccountOrchestratorLoopRunner:
    """``EngineLoopRunner`` backed by the V2 account-execution orchestrator."""

    def __init__(
        self,
        settings: PlatformSettings,
        *,
        engine_names: list[str],
        budgets: tuple[EngineBudget, ...],
        mode: str,
        initial_cash: Decimal,
        instrument_contracts: dict[uuid.UUID, dict[str, object]],
    ) -> None:
        self._settings = settings
        self._engine_names = engine_names
        self._budgets = budgets
        self._mode = mode
        self._initial_cash = initial_cash
        self._contracts = dict(instrument_contracts)
        self._proposal_engines: list[Any] = []
        self._exec_session: Any = None
        self._orchestrator: Any = None
        self._governance_repo: Any = None
        self._v2_bundle: Any = None
        self._clock: Any = None

    @property
    def _session(self) -> Any:
        # Surfaced to the loop's ``engine_session()`` for kill-switch/recovery.
        return self._exec_session

    async def initialize(self) -> None:
        from quant_platform.bootstrap.session.public_api import (
            create_live_session,
            create_paper_session,
        )
        from quant_platform.engines.account.orchestrator import AccountExecutionOrchestrator
        from quant_platform.engines.engine_runner import EngineRunner, RunMode
        from quant_platform.engines.framework.plugins import create_engine_from_plugin
        from quant_platform.engines.multi_engine import MultiEngineRunner
        from quant_platform.infrastructure.repositories.multi_engine_governance import (
            build_multi_engine_governance_repository,
        )
        from quant_platform.infrastructure.support.clock import WallClock
        from quant_platform.infrastructure.v2.postgres import build_v2_repository_bundle
        from quant_platform.services.execution_service.orders.router import DefaultExecutionRouter
        from quant_platform.services.portfolio_service.optimizer import ConstraintAwareOptimizer

        run_mode = RunMode(self._mode)
        self._proposal_engines = [
            create_engine_from_plugin(
                name,
                run_mode=RunMode.PAPER,
                initial_cash=self._initial_cash,
                settings=self._settings,
                instrument_contracts=self._contracts,
            )
            for name in self._engine_names
        ]
        await asyncio.gather(
            *[
                engine.initialize(
                    session_factory=create_engine_runtime_session,
                    scheduler_factory=build_engine_maintenance_scheduler,
                )
                for engine in self._proposal_engines
            ]
        )

        if run_mode == RunMode.LIVE:
            snapshot = await EngineRunner(
                next(iter(self._proposal_engines))._config, self._settings
            )._bootstrap_live_snapshot(self._contracts)
            self._exec_session = create_live_session(
                settings=self._settings,
                initial_snapshot=snapshot,
                instrument_contracts=self._contracts,
            )
        else:
            self._exec_session = create_paper_session(
                settings=self._settings,
                initial_cash=self._initial_cash,
                instrument_contracts=self._contracts or None,
            )

        self._v2_bundle = build_v2_repository_bundle(self._settings)
        self._governance_repo = build_multi_engine_governance_repository(
            self._settings.storage.postgres_dsn
        )
        multi_engine = MultiEngineRunner(
            settings=self._settings,
            budgets=self._budgets,
            governance_repo=self._governance_repo,
        )
        await multi_engine.persist_budgets()

        self._clock = WallClock()
        order_planner = self._exec_session.order_planner
        if order_planner is None:
            raise RuntimeError("orchestrator session is missing order planner")
        self._orchestrator = AccountExecutionOrchestrator(
            multi_engine=multi_engine,
            optimizer=ConstraintAwareOptimizer(),
            order_planner=order_planner,
            approve_ctrl=self._exec_session.approve_ctrl,
            submit_ctrl=self._exec_session.submit_ctrl,
            order_state=self._v2_bundle.order_state,
            execution_router=DefaultExecutionRouter(
                policy=self._exec_session.execution_tactic_policy,
                broker=self._exec_session.trading_broker,
            ),
            risk_repo=self._v2_bundle.risk_models,
            governance_repo=self._governance_repo,
        )

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],  # noqa: ARG002 — orchestrator schedules its own feature jobs
    ) -> CycleResult:
        from quant_platform.application.runtime.state import CycleResult

        account = await self._exec_session.account_broker.sync_account()
        market_prices = {
            position.instrument_id: position.market_price for position in account.positions
        }
        market_prices.update(
            await latest_contract_market_prices(
                exec_session=self._exec_session,
                instrument_contracts=self._contracts,
                existing=market_prices,
                as_of=self._clock.now(),
            )
        )
        risk_model = await self._resolve_risk_model()
        proposals = await asyncio.gather(
            *[
                engine.generate_proposal(market_prices=market_prices)
                for engine in self._proposal_engines
            ]
        )

        strategy_run = self._proposal_engines[0]._strategy_run
        if strategy_run is None:
            raise RuntimeError("proposal engine strategy run is not initialized")
        # Persist the run BEFORE the targets/contributions that carry its id.
        await self._governance_repo.save_strategy_run(strategy_run)
        result = await self._orchestrator.execute(
            proposals=tuple(proposals),
            account=account,
            limits=self._exec_session.risk_limits,
            risk_model=risk_model,
            market_prices=market_prices,
            strategy_run_id=strategy_run.run_id,
            as_of=self._clock.now(),
        )
        await self._governance_repo.save_strategy_run(
            replace(strategy_run, status=RunStatus.COMPLETED, finished_at=self._clock.now())
        )

        return CycleResult(
            signals=[],
            target=None,
            approved=list(result.approved),
            rejected=list(result.rejected),
            submitted_ids=list(result.submitted_ids),
            fills=[],
        )

    async def _resolve_risk_model(self) -> Any:
        risk_models = self._v2_bundle.risk_models
        risk_model = (
            await risk_models.latest_risk_model(as_of=self._clock.now())
            if hasattr(risk_models, "latest_risk_model")
            else None
        )
        if risk_model is not None:
            return risk_model
        from quant_platform.core.domain.portfolio import PortfolioRiskModel

        return PortfolioRiskModel(
            model_id=uuid.uuid4(),
            as_of=self._clock.now(),
            covariance={},
            factor_exposures={},
            scenarios=(),
        )

    async def shutdown(self) -> object:
        summaries = await asyncio.gather(*[engine.shutdown() for engine in self._proposal_engines])
        for summary in summaries:
            log.info(
                "multi_engine.engine_complete",
                engine=summary.engine_name,
                cycles=summary.cycles_completed,
            )
        return summaries


__all__ = ["AccountOrchestratorLoopRunner"]
