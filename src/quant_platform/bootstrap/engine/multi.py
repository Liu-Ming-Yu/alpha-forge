"""V2 multi-engine runtime composition helpers."""

from __future__ import annotations

import asyncio
import json
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.bootstrap.engine.session_wiring import (
    build_engine_maintenance_scheduler,
    create_engine_runtime_session,
)
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.session.public_api import create_live_session, create_paper_session
from quant_platform.engines.market_data.price_seeding import latest_contract_market_prices

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.engines.engine_runner import EngineRunner

log = structlog.get_logger(__name__)


def create_single_engine_runner(
    *,
    settings: PlatformSettings,
    engine_name: str,
    mode: str,
    execution_backend: str,
    initial_cash: Decimal,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> EngineRunner:
    """Create one plugin-backed engine runner from operator strings."""
    from quant_platform.engines.engine_runner import ExecutionBackend, RunMode
    from quant_platform.engines.framework.plugins import create_engine_from_plugin

    return create_engine_from_plugin(
        engine_name,
        run_mode=RunMode(mode),
        initial_cash=initial_cash,
        settings=settings,
        instrument_contracts=instrument_contracts,
        execution_backend=ExecutionBackend(execution_backend),
    )


async def run_multi_engine_v2(
    settings: PlatformSettings,
    *,
    mode: str,
    engine_names: list[str],
    budgets_file: str | None = None,
    cycles: int = 1,
    initial_cash: Decimal = Decimal("50000"),
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
) -> None:
    """Run a V2 multi-engine cycle: proposal engines, orchestrator, submission."""
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

    if not (settings.v2.enabled and settings.v2.account_orchestrator_enabled):
        raise RuntimeError(
            "run-multi-engine requires QP__V2__ENABLED=true and "
            "QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true"
        )

    await verify_postgres_schema(settings)

    run_mode = RunMode(mode)
    budgets = _resolve_budgets(
        budgets_file=budgets_file,
        engine_names=engine_names,
        mode=mode,
    )
    contracts = dict(instrument_contracts or {})
    if run_mode == RunMode.LIVE and not contracts:
        raise ValueError("LIVE mode requires --contracts-file")

    proposal_engines = [
        create_engine_from_plugin(
            name,
            run_mode=RunMode.PAPER,
            initial_cash=initial_cash,
            settings=settings,
            instrument_contracts=contracts,
        )
        for name in engine_names
    ]
    await asyncio.gather(
        *[
            engine.initialize(
                session_factory=create_engine_runtime_session,
                scheduler_factory=build_engine_maintenance_scheduler,
            )
            for engine in proposal_engines
        ]
    )

    if run_mode == RunMode.LIVE:
        snapshot = await EngineRunner(
            next(iter(proposal_engines))._config, settings
        )._bootstrap_live_snapshot(contracts)
        exec_session = create_live_session(
            settings=settings,
            initial_snapshot=snapshot,
            instrument_contracts=contracts,
        )
    else:
        exec_session = create_paper_session(
            settings=settings,
            initial_cash=initial_cash,
            instrument_contracts=contracts or None,
        )

    v2_bundle = build_v2_repository_bundle(settings)
    governance_repo = build_multi_engine_governance_repository(settings.storage.postgres_dsn)
    multi_engine = MultiEngineRunner(
        settings=settings,
        budgets=budgets,
        governance_repo=governance_repo,
    )
    await multi_engine.persist_budgets()

    clock = WallClock()
    order_planner = exec_session.order_planner
    if order_planner is None:
        raise RuntimeError("multi-engine execution session is missing order planner")
    orchestrator = AccountExecutionOrchestrator(
        multi_engine=multi_engine,
        optimizer=ConstraintAwareOptimizer(),
        order_planner=order_planner,
        approve_ctrl=exec_session.approve_ctrl,
        submit_ctrl=exec_session.submit_ctrl,
        order_state=v2_bundle.order_state,
        execution_router=DefaultExecutionRouter(
            policy=exec_session.execution_tactic_policy,
            broker=exec_session.trading_broker,
        ),
        risk_repo=v2_bundle.risk_models,
        governance_repo=governance_repo,
    )

    for cycle_i in range(cycles):
        account = await exec_session.account_broker.sync_account()
        market_prices = {
            position.instrument_id: position.market_price for position in account.positions
        }
        market_prices.update(
            await latest_contract_market_prices(
                exec_session=exec_session,
                instrument_contracts=contracts,
                existing=market_prices,
                as_of=clock.now(),
            )
        )
        risk_model = (
            await v2_bundle.risk_models.latest_risk_model(as_of=clock.now())
            if hasattr(v2_bundle.risk_models, "latest_risk_model")
            else None
        )
        proposals = await asyncio.gather(
            *[engine.generate_proposal(market_prices=market_prices) for engine in proposal_engines]
        )

        if risk_model is None:
            from quant_platform.core.domain.portfolio import PortfolioRiskModel

            risk_model = PortfolioRiskModel(
                model_id=uuid.uuid4(),
                as_of=clock.now(),
                covariance={},
                factor_exposures={},
                scenarios=(),
            )

        strategy_run = proposal_engines[0]._strategy_run
        if strategy_run is None:
            raise RuntimeError("proposal engine strategy run is not initialized")
        result = await orchestrator.execute(
            proposals=tuple(proposals),
            account=account,
            limits=exec_session.risk_limits,
            risk_model=risk_model,
            market_prices=market_prices,
            strategy_run_id=strategy_run.run_id,
            as_of=clock.now(),
        )
        log.info(
            "multi_engine.cycle",
            cycle=cycle_i + 1,
            submitted=len(result.submitted_ids),
            approved=len(result.approved),
            rejected=len(result.rejected),
        )

    summaries = await asyncio.gather(*[engine.shutdown() for engine in proposal_engines])
    for summary in summaries:
        log.info(
            "multi_engine.engine_complete",
            engine=summary.engine_name,
            cycles=summary.cycles_completed,
        )


def _resolve_budgets(
    *,
    budgets_file: str | None,
    engine_names: list[str],
    mode: str,
) -> tuple[Any, ...]:
    if budgets_file is not None:
        return load_budgets(budgets_file, engine_names)
    if len(engine_names) != 1:
        raise ValueError(
            "auto-budget (budgets_file=None) requires exactly one engine; "
            f"got {engine_names}. Provide --budgets-file for multi-engine runs."
        )
    from quant_platform.core.domain.production import EngineBudget

    return (
        EngineBudget(
            engine_name=engine_names[0],
            engine_version="0.1.0",
            run_mode=mode,
            capital_weight=Decimal("1.0"),
            max_gross=Decimal("1.5"),
            max_turnover=Decimal("1.0"),
            enabled=True,
        ),
    )


def load_budgets(
    budgets_file: str,
    engine_names: list[str],
) -> tuple[Any, ...]:
    """Parse a JSON budgets file into a tuple of EngineBudget objects."""
    from quant_platform.core.domain.production import EngineBudget

    with Path(budgets_file).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    budgets = []
    for name in engine_names:
        spec = raw.get(name)
        if spec is None:
            raise ValueError(f"No budget entry for engine '{name}' in {budgets_file}")
        budgets.append(
            EngineBudget(
                engine_name=name,
                engine_version=spec.get("engine_version", "0.1.0"),
                capital_weight=Decimal(str(spec["capital_weight"])),
                max_gross=Decimal(str(spec["max_gross"])),
                max_turnover=Decimal(str(spec["max_turnover"])),
                enabled=spec.get("enabled", True),
                run_mode=spec.get("run_mode", "live"),
            )
        )
    return tuple(budgets)


__all__ = [
    "create_single_engine_runner",
    "latest_contract_market_prices",
    "load_budgets",
    "run_multi_engine_v2",
]
