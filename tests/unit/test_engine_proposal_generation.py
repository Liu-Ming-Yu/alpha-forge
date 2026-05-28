"""Unit tests for EngineRunner.generate_proposal() and V2 run_cycle guard."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quant_platform.config import PlatformSettings, V2Settings
from quant_platform.core.domain.portfolio import PortfolioTarget, RiskLimits
from quant_platform.core.domain.production import EngineTargetProposal
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.engine_runner import (
    EngineConfig,
    EngineRunner,
    ExecutionBackend,
    RunMode,
)
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.session import CycleResult

_NOW = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)
_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()


def _make_target(run_id: uuid.UUID) -> PortfolioTarget:
    return PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=run_id,
        as_of=_NOW,
        regime_id=uuid.uuid4(),
        weights={_INST_A: Decimal("0.30"), _INST_B: Decimal("0.20")},
        cash_target_weight=Decimal("0.50"),
        construction_notes=["test_note"],
    )


def _make_strategy_run(engine_name: str) -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name=engine_name,
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
    )


def _make_regime() -> MagicMock:
    regime = MagicMock()
    regime.regime_label = MagicMock()
    regime.regime_label.value = "risk_on"
    regime.regime_id = uuid.uuid4()
    return regime


def _patch_session(runner: EngineRunner, run_id: uuid.UUID) -> MagicMock:
    """Wire a fake session onto an uninitialized runner."""
    target = _make_target(run_id)
    regime = _make_regime()

    signal_ctrl = AsyncMock()
    signal_ctrl.generate = AsyncMock(return_value=[MagicMock(), MagicMock()])

    portfolio_ctrl = AsyncMock()
    portfolio_ctrl.build = AsyncMock(return_value=target)

    regime_detector = AsyncMock()
    regime_detector.detect = AsyncMock(return_value=regime)

    account_broker = AsyncMock()
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot

    account_broker.sync_account = AsyncMock(
        return_value=AccountSnapshot(
            snapshot_id=uuid.uuid4(),
            as_of=_NOW,
            settled_cash=Decimal("50000"),
            unsettled_cash=Decimal("0"),
            reserved_cash=Decimal("0"),
            available_cash=Decimal("50000"),
            net_asset_value=Decimal("50000"),
            positions=(),
        )
    )

    clock = FakeClock(_NOW)

    limits = RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=run_id,
        effective_from=_NOW,
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.40"),
        max_gross_exposure=Decimal("0.80"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.15"),
    )

    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()

    from quant_platform.core.domain.instruments import AssetClass, Instrument
    from quant_platform.services.data_service.reference.contract_master import ContractMaster

    contract_master = ContractMaster(
        [
            Instrument(
                instrument_id=_INST_A,
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                exchange="NASDAQ",
                currency="USD",
                active=True,
            ),
            Instrument(
                instrument_id=_INST_B,
                symbol="MSFT",
                asset_class=AssetClass.EQUITY,
                exchange="NASDAQ",
                currency="USD",
                active=True,
            ),
        ]
    )

    session = MagicMock()
    session.clock = clock
    session.signal_ctrl = signal_ctrl
    session.portfolio_ctrl = portfolio_ctrl
    session.regime_detector = regime_detector
    session.account_broker = account_broker
    session.risk_limits = limits
    session.event_bus = event_bus
    session.contract_master = contract_master
    session.bar_store = None
    session.feature_repo = MagicMock()
    session._state_hydrated = True

    runner._session = session
    runner._strategy_run = _make_strategy_run(runner._config.engine_name)
    return session


@pytest.mark.asyncio
async def test_generate_proposal_returns_correct_weights() -> None:
    """generate_proposal() returns an EngineTargetProposal with the portfolio target weights."""
    settings = PlatformSettings(_env_file=None)
    config = EngineConfig(engine_name="test_engine", run_mode=RunMode.PAPER)
    runner = EngineRunner(config=config, settings=settings)
    session = _patch_session(runner, uuid.uuid4())

    with patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock):
        proposal = await runner.generate_proposal(feature_data={_INST_A: {"momentum": 0.5}})

    assert isinstance(proposal, EngineTargetProposal)
    assert proposal.engine_name == "test_engine"
    assert proposal.weights[_INST_A] == Decimal("0.30")
    assert proposal.weights[_INST_B] == Decimal("0.20")
    assert proposal.cash_target_weight == Decimal("0.50")
    assert "test_note" in proposal.notes
    assert proposal.promotion_state == "paper"

    # Event should have been published
    session.event_bus.publish.assert_called_once()
    published_event = session.event_bus.publish.call_args[0][0]
    assert published_event.proposal_id == proposal.proposal_id
    assert published_event.weight_count == 2


@pytest.mark.asyncio
async def test_generate_proposal_increments_cycle_counter() -> None:
    settings = PlatformSettings(_env_file=None)
    config = EngineConfig(engine_name="test_engine", run_mode=RunMode.PAPER)
    runner = EngineRunner(config=config, settings=settings)
    _patch_session(runner, uuid.uuid4())

    assert runner._result.cycles_completed == 0
    with patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock):
        await runner.generate_proposal(feature_data={})
    assert runner._result.cycles_completed == 1


@pytest.mark.asyncio
async def test_generate_proposal_fails_closed_on_stale_feature_data() -> None:
    """When require_feature_datasets=True and snapshot is stale, raises RuntimeError."""
    settings = PlatformSettings(
        _env_file=None,
        v2=V2Settings(
            enabled=True,
            require_feature_datasets=True,
            max_feature_age_seconds=3600,
        ),
    )
    config = EngineConfig(engine_name="test_engine", run_mode=RunMode.PAPER)
    runner = EngineRunner(config=config, settings=settings)
    session = _patch_session(runner, uuid.uuid4())

    stale_as_of = datetime(2026, 1, 14, 0, 0, tzinfo=UTC)  # ~33h before _NOW

    stale_snapshot = MagicMock()
    stale_snapshot.as_of = stale_as_of
    stale_snapshot.dataset.dataset_id = uuid.uuid4()

    dataset_catalog = AsyncMock()
    session.dataset_catalog = dataset_catalog

    with (
        patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock),
        patch(
            "quant_platform.services.research_service.feature_quality.snapshot.load_feature_snapshot",
            new_callable=AsyncMock,
            return_value=stale_snapshot,
        ),
        pytest.raises(RuntimeError, match="feature snapshot is stale"),
    ):
        await runner.generate_proposal(feature_data={})


@pytest.mark.asyncio
async def test_run_cycle_v2_mode_stops_before_submission() -> None:
    """run_cycle() in PAPER mode with orchestrator enabled returns proposal, no broker calls."""
    settings = PlatformSettings(
        _env_file=None,
        v2=V2Settings(enabled=True, account_orchestrator_enabled=True),
    )
    config = EngineConfig(engine_name="test_engine", run_mode=RunMode.PAPER)
    runner = EngineRunner(config=config, settings=settings)
    session = _patch_session(runner, uuid.uuid4())

    with patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock):
        result = await runner.run_cycle(feature_data={_INST_A: {"momentum": 0.5}})

    assert isinstance(result, CycleResult)
    assert result.proposal is not None
    assert isinstance(result.proposal, EngineTargetProposal)
    # No orders submitted in V2 proposal mode
    assert result.submitted_ids == []
    assert result.approved == []
    # Event published for the proposal
    session.event_bus.publish.assert_called_once()


@pytest.mark.asyncio
async def test_run_cycle_v1_mode_submits_normally() -> None:
    """run_cycle() without V2 orchestrator goes through the normal submission path."""
    settings = PlatformSettings(_env_file=None)  # V2 disabled
    config = EngineConfig(engine_name="test_engine", run_mode=RunMode.PAPER)
    runner = EngineRunner(config=config, settings=settings)
    _patch_session(runner, uuid.uuid4())

    cycle_result = CycleResult(
        signals=[MagicMock()],
        target=MagicMock(),
        approved=[MagicMock()],
        rejected=[],
        submitted_ids=[uuid.uuid4()],
        fills=[],
    )

    with (
        patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock),
        patch(
            "quant_platform.engines.engine_runner.run_strategy_cycle",
            new_callable=AsyncMock,
            return_value=cycle_result,
        ) as mock_cycle,
    ):
        result = await runner.run_cycle(feature_data={})

    mock_cycle.assert_called_once()
    assert result.proposal is None
    assert len(result.submitted_ids) == 1


@pytest.mark.asyncio
async def test_run_cycle_ib_paper_fails_closed_on_empty_feature_data() -> None:
    settings = PlatformSettings(_env_file=None)
    config = EngineConfig(
        engine_name="test_engine",
        run_mode=RunMode.PAPER,
        execution_backend=ExecutionBackend.IB_PAPER,
        required_features=("momentum",),
    )
    runner = EngineRunner(config=config, settings=settings)
    _patch_session(runner, uuid.uuid4())

    with (
        patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock),
        patch(
            "quant_platform.engines.engine_runner.run_strategy_cycle",
            new_callable=AsyncMock,
        ) as mock_cycle,
        pytest.raises(DataStalenessError, match="no feature data available"),
    ):
        await runner.run_cycle(feature_data={})

    mock_cycle.assert_not_called()


@pytest.mark.asyncio
async def test_run_cycle_ib_paper_merges_contract_market_prices() -> None:
    settings = PlatformSettings(_env_file=None)
    config = EngineConfig(
        engine_name="test_engine",
        run_mode=RunMode.PAPER,
        execution_backend=ExecutionBackend.IB_PAPER,
        instrument_contracts={
            _INST_A: {"last_close": "123.45"},
            _INST_B: {"last_close": "67.89"},
        },
        required_features=("momentum",),
    )
    runner = EngineRunner(config=config, settings=settings)
    _patch_session(runner, uuid.uuid4())
    cycle_result = CycleResult(
        signals=[],
        target=None,
        approved=[],
        rejected=[],
        submitted_ids=[],
        fills=[],
    )

    with (
        patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock),
        patch(
            "quant_platform.engines.engine_runner.run_strategy_cycle",
            new_callable=AsyncMock,
            return_value=cycle_result,
        ) as mock_cycle,
    ):
        result = await runner.run_cycle(
            feature_data={
                _INST_A: {"momentum": 0.5},
                _INST_B: {"momentum": 0.4},
            },
            market_prices={_INST_A: Decimal("200")},
        )

    assert result is cycle_result
    prices = mock_cycle.call_args.kwargs["market_prices"]
    assert prices[_INST_A] == Decimal("200")
    assert prices[_INST_B] == Decimal("67.89")


@pytest.mark.asyncio
async def test_run_cycle_ib_paper_fails_closed_on_missing_reference_price() -> None:
    settings = PlatformSettings(_env_file=None)
    config = EngineConfig(
        engine_name="test_engine",
        run_mode=RunMode.PAPER,
        execution_backend=ExecutionBackend.IB_PAPER,
        instrument_contracts={_INST_A: {"last_close": "123.45"}},
        required_features=("momentum",),
    )
    runner = EngineRunner(config=config, settings=settings)
    _patch_session(runner, uuid.uuid4())

    with (
        patch("quant_platform.session.hydrate_session_state", new_callable=AsyncMock),
        patch(
            "quant_platform.engines.engine_runner.run_strategy_cycle",
            new_callable=AsyncMock,
        ) as mock_cycle,
        pytest.raises(DataStalenessError, match="missing positive reference prices"),
    ):
        await runner.run_cycle(
            feature_data={
                _INST_A: {"momentum": 0.5},
                _INST_B: {"momentum": 0.4},
            }
        )

    mock_cycle.assert_not_called()
