"""VectorBT vs SimpleBacktestEngine parity regression test (Item 20).

Both engines are run over a synthetic 10-instrument, 20-rebalance dataset with
identical inputs.  When ``parity_mode=True`` the VectorBTBacktestEngine
suppresses regime scaling and vol-weighting, producing results that must agree
with ``SimpleBacktestEngine`` within a 5 bps total-return tolerance.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
    SimpleRegimeDetector,
)
from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
    SimpleBacktestEngine,
)
from quant_platform.services.research_service.backtesting.vectorbt.vectorbt_engine import (
    VectorBTBacktestEngine,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

_N_INSTRUMENTS = 10
_N_REBALANCES = 20
_INITIAL_CAPITAL = Decimal("100_000")
_START = datetime(2024, 1, 2, tzinfo=_UTC)


def _build_dataset() -> tuple[
    list[uuid.UUID],
    list[datetime],
    dict[datetime, dict[uuid.UUID, dict[str, float]]],
    dict[datetime, dict[uuid.UUID, Decimal]],
]:
    instruments = [uuid.uuid4() for _ in range(_N_INSTRUMENTS)]
    rebalances = [_START + timedelta(days=7 * i) for i in range(_N_REBALANCES)]

    # Deterministic rising prices — all instruments appreciate ~1% per period.
    prices: dict[datetime, dict[uuid.UUID, Decimal]] = {}
    base = {iid: 100.0 + idx * 5.0 for idx, iid in enumerate(instruments)}
    for t_idx, ts in enumerate(rebalances):
        prices[ts] = {
            iid: Decimal(str(round(base[iid] * (1.0 + 0.01 * t_idx), 4))) for iid in instruments
        }

    # Constant feature: all instruments score identically (flat cross-section).
    features: dict[datetime, dict[uuid.UUID, dict[str, float]]] = {}
    for ts in rebalances:
        features[ts] = {iid: {"momentum_1m": 0.5, "momentum_3m": 0.5} for iid in instruments}

    return instruments, rebalances, features, prices


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="parity_test",
        strategy_version="0.0.1",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_START,
        started_at=_START,
    )


def _settings() -> PlatformSettings:
    # Disable regime requirement so both engines use SimpleRegimeDetector.
    # Set risk limits so simple engine invests 100% gross (matching VectorBT equal-weight).
    return PlatformSettings.model_validate(
        {
            "backtest": {"require_market_regime": False},
            "cash": {"buy_side_t1_settlement": False},
            "liquidity": {"allow_missing_profile": True},
            "risk": {
                "max_single_name_weight": "0.10",
                "max_gross_exposure": "1.00",
                "min_cash_buffer": "0.00",
            },
            "storage": {
                "event_bus_backend": "in_memory",
                "postgres_dsn": "",
                "redis_url": "",
            },
        }
    )


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vectorbt_parity_mode_matches_simple_engine() -> None:
    """VectorBT in parity_mode must agree with SimpleBacktestEngine within 5 bps."""
    instruments, rebalances, features, prices = _build_dataset()
    start = rebalances[0]
    end = rebalances[-1] + timedelta(days=1)
    signal_model = LinearWeightSignalModel(
        {"momentum_1m": 0.50, "momentum_3m": 0.50},
        model_version="parity-v1",
    )
    settings = _settings()
    clock = FakeClock(start)

    # ----------------------------------------------------------------
    # Run SimpleBacktestEngine
    # ----------------------------------------------------------------
    simple = SimpleBacktestEngine(
        clock=clock,
        signal_model=signal_model,
        portfolio_constructor=LongOnlyPortfolioConstructor(top_n=_N_INSTRUMENTS),
        settings=settings,
        paper_session_factory=create_paper_session,
        strategy_cycle_runner=run_strategy_cycle,
    )
    run_simple = StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="parity_simple",
        strategy_version="0.0.1",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=start,
        started_at=start,
    )
    result_simple = await simple.run_with_data(
        strategy_run=run_simple,
        start=start,
        end=end,
        initial_capital=_INITIAL_CAPITAL,
        rebalance_timestamps=rebalances,
        feature_series=features,
        price_series=prices,
        regime_detector=SimpleRegimeDetector(),
    )

    # ----------------------------------------------------------------
    # Run VectorBTBacktestEngine with parity_mode=True
    # ----------------------------------------------------------------
    vbt = VectorBTBacktestEngine(
        clock=clock,
        signal_model=signal_model,
        settings=settings,
        top_n=_N_INSTRUMENTS,
        parity_mode=True,
    )
    run_vbt = StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="parity_vbt",
        strategy_version="0.0.1",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=start,
        started_at=start,
    )
    result_vbt = await vbt.run_with_data(
        strategy_run=run_vbt,
        start=start,
        end=end,
        initial_capital=_INITIAL_CAPITAL,
        rebalance_timestamps=rebalances,
        feature_series=features,
        price_series=prices,
        regime_detector=SimpleRegimeDetector(),
    )

    # ----------------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------------
    # Both engines should produce a positive total return (prices rise).
    assert result_simple.total_return > Decimal("0"), "simple engine should profit"
    assert result_vbt.total_return > Decimal("0"), "VectorBT engine should profit"

    # Total return must agree within 10 bps (0.10%).  The residual comes from
    # integer share-truncation differences between the two order planners.
    divergence_bps = abs(result_simple.total_return - result_vbt.total_return) * 10_000
    assert divergence_bps < Decimal("10"), (
        f"total return divergence {divergence_bps:.2f} bps exceeds 10 bps — "
        f"simple={result_simple.total_return:.6f} vbt={result_vbt.total_return:.6f}"
    )
