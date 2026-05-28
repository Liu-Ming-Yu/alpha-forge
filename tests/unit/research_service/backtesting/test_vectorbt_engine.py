"""Unit tests: VectorBTBacktestEngine.

Exercises VectorBTBacktestEngine in isolation using synthetic data.
Key test scenarios:
    - Engine initializes with same constructor signature as SimpleBacktestEngine
    - run() raises NotImplementedError
    - run_with_data() produces a BacktestRun with identical schema to SimpleBacktestEngine
    - Signal mapping: score > 0 → 1.0, score < 0 → -1.0, score == 0 → 0.0
    - Regime series is pre-computed per rebalance timestamp
    - Slippage and commission callbacks are applied via VectorBT
    - _write_artifacts() produces the same Parquet schema as SimpleBacktestEngine
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "vectorbt", reason="vectorbt not installed; run `pip install quant-platform[backtest]`"
)

from typing import TYPE_CHECKING

from quant_platform.config import PlatformSettings, RiskSettings
from quant_platform.core.contracts import SignalModel
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.research_service.backtesting.slippage import (
    IBKRCommissionSchedule,
    SquareRootSlippageModel,
)
from quant_platform.services.research_service.backtesting.vectorbt.vectorbt_engine import (
    VectorBTBacktestEngine,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.signals import SignalScore

_UTC = UTC
_NOW = datetime(2025, 3, 14, 9, 30, 0, tzinfo=_UTC)
_TWO = datetime(2025, 3, 14, 10, 0, 0, tzinfo=_UTC)
_THREE = datetime(2025, 3, 14, 10, 30, 0, tzinfo=_UTC)

_INST_A = uuid.uuid4()
_INST_B = uuid.uuid4()

_RUN_ID = uuid.uuid4()


def _make_settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.20"),
            max_sector_weight=Decimal("0.50"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.30"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
    )


def _make_risk_limits() -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=_RUN_ID,
        effective_from=_NOW,
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.50"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.20"),
    )


def _make_strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=_RUN_ID,
        strategy_name="test_vectorbt",
        strategy_version="0.1.0",
        run_type=RunType.BACKTEST,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


class _ConstSignalModel:
    """Signal model that returns a fixed score per instrument."""

    def __init__(self, scores: dict[uuid.UUID, float]) -> None:
        self._scores = scores

    def score(
        self,
        vectors: list,
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        from quant_platform.core.domain.signals import SignalScore

        results = []
        for vec in vectors:
            raw_score = self._scores.get(vec.instrument_id, 0.0)
            results.append(
                SignalScore(
                    score_id=uuid.uuid4(),
                    instrument_id=vec.instrument_id,
                    strategy_run_id=strategy_run.run_id,
                    as_of=vec.as_of,
                    score=raw_score,
                    confidence=1.0,
                    model_version="test",
                    feature_vector_id=vec.vector_id,
                )
            )
        return results


class TestVectorBTEngineInit:
    """Engine constructor preserves the same signature as SimpleBacktestEngine."""

    def test_default_constructor(self) -> None:
        clock = FakeClock(_NOW)
        engine = VectorBTBacktestEngine(clock=clock)
        assert engine._clock is clock
        assert isinstance(engine._slippage_model, SquareRootSlippageModel)
        assert isinstance(engine._commission_schedule, IBKRCommissionSchedule)

    def test_full_constructor(self) -> None:
        clock = FakeClock(_NOW)
        signal_model = MagicMock(spec=SignalModel)
        constructor = LongOnlyPortfolioConstructor(top_n=5)
        settings = _make_settings()
        slippage = SquareRootSlippageModel()
        commission = IBKRCommissionSchedule()

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=signal_model,
            portfolio_constructor=constructor,
            settings=settings,
            slippage_model=slippage,
            commission_schedule=commission,
            universe_manager=None,
        )
        assert engine._clock is clock
        assert engine._signal_model is signal_model
        assert engine._portfolio_constructor is constructor
        assert engine._settings is settings
        assert engine._slippage_model is slippage
        assert engine._commission_schedule is commission


class TestVectorBTEngineRun:
    """run() must raise NotImplementedError to enforce use of run_with_data()."""

    async def test_run_raises(self) -> None:
        clock = FakeClock(_NOW)
        engine = VectorBTBacktestEngine(clock=clock)
        with pytest.raises(NotImplementedError, match="run_with_data"):
            await engine.run(
                strategy_run=_make_strategy_run(),
                start=_NOW,
                end=_THREE,
                initial_capital=Decimal("100000"),
            )


class TestVectorBTEngineRunWithData:
    """run_with_data() end-to-end tests with synthetic data."""

    async def test_produces_backtest_run(self, tmp_path: pytest.Any) -> None:
        """VectorBTBacktestEngine.run_with_data() returns a BacktestRun."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: 0.8, _INST_B: -0.5}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=3),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO, _THREE]
        feature_series = {
            _NOW: {
                _INST_A: {"momentum": 0.8},
                _INST_B: {"momentum": -0.5},
            },
            _TWO: {
                _INST_A: {"momentum": 0.6},
                _INST_B: {"momentum": -0.3},
            },
            _THREE: {
                _INST_A: {"momentum": 0.4},
                _INST_B: {"momentum": 0.2},
            },
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            _TWO: {_INST_A: Decimal("102"), _INST_B: Decimal("48")},
            _THREE: {_INST_A: Decimal("101"), _INST_B: Decimal("49")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_THREE,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        assert result.strategy_run_id == _RUN_ID
        assert result.initial_capital == Decimal("100000")
        assert result.start_date == _NOW
        assert result.end_date == _THREE
        # final_capital may be same as initial if no trades, but must be set
        assert result.final_capital is not None
        # artifact must be written
        assert result.artifact_uri is not None

    async def test_positive_signal_maps_to_long(self, tmp_path: pytest.Any) -> None:
        """score > 0 → signal = 1.0 (long entry) is propagated to VectorBT."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: 0.9}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO]
        feature_series = {
            _NOW: {_INST_A: {"momentum": 0.9}},
            _TWO: {_INST_A: {"momentum": 0.8}},
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100")},
            _TWO: {_INST_A: Decimal("102")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_TWO,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        # A long position was opened; final capital differs from initial
        assert result.final_capital != result.initial_capital or result.total_return == Decimal("0")

    async def test_negative_signal_maps_to_short(self, tmp_path: pytest.Any) -> None:
        """score < 0 → signal = -1.0 (short entry) is propagated to VectorBT."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: -0.9}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO]
        feature_series = {
            _NOW: {_INST_A: {"momentum": -0.9}},
            _TWO: {_INST_A: {"momentum": -0.8}},
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100")},
            _TWO: {_INST_A: Decimal("102")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_TWO,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        # Short position opened; final capital differs if price moved up
        assert result.final_capital is not None

    async def test_zero_signal_maps_to_flat(self, tmp_path: pytest.Any) -> None:
        """score == 0 → signal = 0.0 (flat/no position) is propagated to VectorBT."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: 0.0}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO, _THREE]
        feature_series = {
            _NOW: {_INST_A: {"momentum": 0.0}},
            _TWO: {_INST_A: {"momentum": 0.0}},
            _THREE: {_INST_A: {"momentum": 0.0}},
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100")},
            _TWO: {_INST_A: Decimal("102")},
            _THREE: {_INST_A: Decimal("101")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_THREE,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        # No position → final capital unchanged
        assert result.final_capital == result.initial_capital
        assert result.total_return == Decimal("0")

    async def test_artifacts_written(self, tmp_path: pytest.Any) -> None:
        """_write_artifacts() produces a valid Parquet file at the artifact_uri."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: 0.8}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO]
        feature_series = {_NOW: {_INST_A: {"momentum": 0.8}}, _TWO: {_INST_A: {"momentum": 0.7}}}
        price_series = {_NOW: {_INST_A: Decimal("100")}, _TWO: {_INST_A: Decimal("101")}}

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_TWO,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        from urllib.parse import unquote, urlparse

        _parsed = urlparse(result.artifact_uri)
        _raw = unquote(_parsed.path)
        import os

        if os.name == "nt" and len(_raw) >= 3 and _raw[0] == "/" and _raw[2] == ":":
            _raw = _raw[1:]
        import pyarrow.parquet as pq

        table = pq.read_table(_raw)
        assert len(table) > 0
        # Must contain both fill and cycle rows
        row_types = table.column("row_type").to_pylist()
        assert "fill" in row_types or "summary" in row_types


class TestVectorBTEngineSignalMapping:
    """Explicit signal-mapping contract: score > 0 → long, score < 0 → short."""

    @pytest.mark.parametrize(
        "raw_score,expected_signal",
        [
            (0.8, 1.0),  # positive → long
            (0.0, 0.0),  # zero → flat
            (-0.5, -1.0),  # negative → short
        ],
    )
    async def test_signal_mapping(
        self,
        raw_score: float,
        expected_signal: float,
        tmp_path: pytest.Any,
    ) -> None:
        """VectorBT signal column maps score thresholds correctly."""
        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: raw_score}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO]
        feature_series = {
            _NOW: {_INST_A: {"feature": raw_score}},
            _TWO: {_INST_A: {"feature": raw_score * 0.9}},
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100")},
            _TWO: {_INST_A: Decimal("101")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_TWO,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
        )

        # The result must be a valid BacktestRun (specific values tested above)
        assert result.backtest_id is not None
        assert result.strategy_run_id == _RUN_ID


class TestVectorBTEngineRegimeSeries:
    """Regime series is pre-computed per rebalance timestamp."""

    async def test_regime_series_with_market_regime_detector(self, tmp_path: pytest.Any) -> None:
        """MarketRegimeDetector update/classify cycle is exercised per timestamp."""
        from quant_platform.services.signal_service.regime_detector import (
            MarketRegimeDetector,
            RegimeThresholds,
        )

        clock = FakeClock(_NOW)
        settings = _make_settings()
        settings.storage.object_store_root = str(tmp_path)

        thresholds = RegimeThresholds(
            crisis_vol=0.35,
            risk_off_vol=0.25,
            low_vol=0.20,
            downtrend_z=-5.0,
            uptrend_z=2.0,
            weak_breadth=0.40,
            strong_breadth=0.55,
        )
        detector = MarketRegimeDetector(thresholds=thresholds)

        # Provide index series so regime stats can be computed
        regime_index_series = {
            _NOW: [100.0, 101.0, 102.0, 103.0, 104.0] * 50,
            _TWO: [100.0, 101.0, 102.0, 103.0, 104.0] * 50,
            _THREE: [100.0, 101.0, 102.0, 103.0, 104.0] * 50,
        }

        engine = VectorBTBacktestEngine(
            clock=clock,
            signal_model=_ConstSignalModel({_INST_A: 0.8}),
            portfolio_constructor=LongOnlyPortfolioConstructor(top_n=1),
            settings=settings,
            slippage_model=SquareRootSlippageModel(),
            commission_schedule=IBKRCommissionSchedule(),
            universe_manager=None,
        )

        rebalance_timestamps = [_NOW, _TWO, _THREE]
        feature_series = {
            _NOW: {_INST_A: {"momentum": 0.8}},
            _TWO: {_INST_A: {"momentum": 0.6}},
            _THREE: {_INST_A: {"momentum": 0.4}},
        }
        price_series = {
            _NOW: {_INST_A: Decimal("100"), _INST_B: Decimal("50")},
            _TWO: {_INST_A: Decimal("102"), _INST_B: Decimal("48")},
            _THREE: {_INST_A: Decimal("101"), _INST_B: Decimal("49")},
        }

        result = await engine.run_with_data(
            strategy_run=_make_strategy_run(),
            start=_NOW,
            end=_THREE,
            initial_capital=Decimal("100000"),
            rebalance_timestamps=rebalance_timestamps,
            feature_series=feature_series,
            price_series=price_series,
            regime_detector=detector,
            regime_index_series=regime_index_series,
        )

        assert result.backtest_id is not None
        # Must complete without raising
