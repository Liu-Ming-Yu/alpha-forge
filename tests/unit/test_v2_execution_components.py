from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings, V2Settings
from quant_platform.core.domain.orders import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_platform.core.domain.portfolio import PortfolioRiskModel, PortfolioTarget, RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.production import EngineBudget, EngineTargetProposal
from quant_platform.engines.account.orchestrator import AccountExecutionOrchestrator
from quant_platform.engines.multi_engine import MultiEngineRunner
from quant_platform.infrastructure.repositories.multi_engine_governance import (
    InMemoryMultiEngineGovernanceRepository,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.infrastructure.v2.postgres import build_v2_repository_bundle
from quant_platform.infrastructure.v2.state import (
    InMemoryOrderStateStore,
    InMemoryPortfolioRiskModelRepository,
)
from quant_platform.services.execution_service.orders.router import DefaultExecutionRouter
from quant_platform.services.portfolio_service.optimizer import ConstraintAwareOptimizer
from quant_platform.services.portfolio_service.order_planner import PortfolioTargetOrderPlanner


def _account(as_of: datetime) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=as_of,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _limits(as_of: datetime, run_id: uuid.UUID) -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=run_id,
        effective_from=as_of,
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.40"),
        max_gross_exposure=Decimal("0.80"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.10"),
    )


def test_v2_repository_bundle_refuses_in_memory_when_v2_live_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QP__STORAGE__POSTGRES_DSN", raising=False)
    settings = PlatformSettings(
        _env_file=None,
        v2=V2Settings(enabled=True, account_orchestrator_enabled=True),
    )

    with pytest.raises(RuntimeError, match="requires QP__STORAGE__POSTGRES_DSN"):
        build_v2_repository_bundle(settings)


def test_constraint_aware_optimizer_applies_single_name_and_cash_caps() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    run_id = uuid.uuid4()
    first = uuid.uuid4()
    second = uuid.uuid4()
    target = PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=run_id,
        as_of=as_of,
        regime_id=uuid.uuid4(),
        weights={first: Decimal("0.50"), second: Decimal("0.30")},
        cash_target_weight=Decimal("0.20"),
    )
    risk_model = PortfolioRiskModel(
        model_id=uuid.uuid4(),
        as_of=as_of,
        covariance={
            (first, first): Decimal("0.0400"),
            (second, second): Decimal("0.0100"),
        },
        factor_exposures={first: {"momentum": Decimal("1")}, second: {"momentum": Decimal("-1")}},
    )

    result = ConstraintAwareOptimizer().optimize(
        target,
        _account(as_of),
        _limits(as_of, run_id),
        risk_model,
    )

    assert max(result.target.weights.values()) <= Decimal("0.20")
    assert result.target.cash_target_weight >= Decimal("0.20")
    assert result.risk_snapshot.passed
    assert "covariance_scaled" in result.binding_constraints


def test_constraint_aware_optimizer_applies_factor_caps() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    run_id = uuid.uuid4()
    first = uuid.uuid4()
    second = uuid.uuid4()
    third = uuid.uuid4()
    target = PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=run_id,
        as_of=as_of,
        regime_id=uuid.uuid4(),
        weights={
            first: Decimal("0.30"),
            second: Decimal("0.30"),
            third: Decimal("0.10"),
        },
        cash_target_weight=Decimal("0.30"),
    )
    risk_model = PortfolioRiskModel(
        model_id=uuid.uuid4(),
        as_of=as_of,
        covariance={
            (first, first): Decimal("0.0100"),
            (second, second): Decimal("0.0100"),
            (third, third): Decimal("0.0100"),
        },
        factor_exposures={
            first: {"sector:technology": Decimal("1")},
            second: {"sector:technology": Decimal("1")},
            third: {"sector:utilities": Decimal("1")},
        },
    )

    result = ConstraintAwareOptimizer().optimize(
        target,
        _account(as_of),
        RiskLimits(
            limits_id=uuid.uuid4(),
            strategy_run_id=run_id,
            effective_from=as_of,
            max_single_name_weight=Decimal("0.30"),
            max_sector_weight=Decimal("0.40"),
            max_gross_exposure=Decimal("0.80"),
            max_daily_turnover=Decimal("0.30"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.10"),
        ),
        risk_model,
    )

    assert result.risk_snapshot.factor_exposures["sector:technology"] <= Decimal("0.40")
    assert "factor_cap" in result.binding_constraints


def test_default_execution_router_selects_auction_and_passive_tactics() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    router = DefaultExecutionRouter()
    moc = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        side=OrderSide.SELL,
        quantity=1,
        order_type=OrderType.MOC,
        time_in_force=TimeInForce.DAY,
        created_at=as_of,
    )

    assert router.route(moc).tactic.value == "close_auction_moc"


@pytest.mark.asyncio
async def test_account_execution_orchestrator_writes_order_state_events() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    run_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    governance = InMemoryMultiEngineGovernanceRepository()
    order_state = InMemoryOrderStateStore()
    risk_repo = InMemoryPortfolioRiskModelRepository()
    runner = MultiEngineRunner(
        PlatformSettings(
            _env_file=None,
            allow_dev_defaults=True,
            v2=V2Settings(enabled=True, account_orchestrator_enabled=True),
        ),
        governance_repo=governance,
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("0.20"),
            ),
        ),
    )
    orchestrator = AccountExecutionOrchestrator(
        multi_engine=runner,
        optimizer=ConstraintAwareOptimizer(),
        order_planner=PortfolioTargetOrderPlanner(FakeClock(as_of)),
        approve_ctrl=_ApproveAll(),
        submit_ctrl=_SubmitAll(),
        order_state=order_state,
        execution_router=DefaultExecutionRouter(),
        risk_repo=risk_repo,
        governance_repo=governance,
    )

    result = await orchestrator.execute(
        proposals=(
            EngineTargetProposal(
                proposal_id=uuid.uuid4(),
                engine_name="cross_sectional_equity_v1",
                engine_version="0.1.0",
                run_mode="paper",
                strategy_run_id=run_id,
                as_of=as_of,
                weights={instrument_id: Decimal("0.50")},
                cash_target_weight=Decimal("0.50"),
            ),
        ),
        account=_account(as_of),
        limits=_limits(as_of, run_id),
        risk_model=PortfolioRiskModel(
            model_id=uuid.uuid4(),
            as_of=as_of,
            covariance={(instrument_id, instrument_id): Decimal("0.01")},
            factor_exposures={},
        ),
        market_prices={instrument_id: Decimal("100")},
        strategy_run_id=run_id,
        as_of=as_of,
    )

    assert len(result.submitted_ids) == 1
    events = await order_state.list_events(result.submitted_ids[0])
    assert [event.event_type.value for event in events] == [
        "created",
        "approved",
        "routed",
        "acknowledged",
    ]
    assert await governance.list_order_allocations(result.submitted_ids[0])


class _ApproveAll:
    async def approve(
        self,
        intents: list[OrderIntent],
        _account: AccountSnapshot,
    ) -> tuple[list[OrderIntent], list[OrderIntent]]:
        return intents, []


class _SubmitAll:
    async def submit(
        self,
        approved: list[OrderIntent],
        account: AccountSnapshot | None = None,
    ) -> list[uuid.UUID]:
        return [intent.order_id for intent in approved]
