"""Integration test: bar data → FeatureBundle → signals → PortfolioTarget.

End-to-end smoke test that exercises the full Phase 2 research pipeline:
  1. Synthetic bar data generation
  2. build_feature_bundle → FeatureBundle (cross-sectional normalization)
  3. LinearWeightSignalModel scoring via GenerateSignalsControllerImpl
  4. MarketRegimeDetector.compute_stats + classify
  5. VolTargetedPortfolioConstructor.build_targets → PortfolioTarget

No mocks — only lightweight in-process fakes (null EventBus).
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)
from quant_platform.services.research_service.features.cross_section.cross_section import (
    build_feature_bundle,
)
from quant_platform.services.signal_service.controllers import (
    GenerateSignalsControllerImpl,
)
from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)
_RUN_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NullEventBus:
    """Discards all events; satisfies the EventBus Protocol."""

    async def publish(self, event: object) -> None:  # noqa: D401
        pass

    async def subscribe(self, event_type: type, handler: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_bar_data(
    n_instruments: int,
    n_bars: int,
    seed: int = 42,
) -> dict[uuid.UUID, list[float]]:
    rng = random.Random(seed)
    data: dict[uuid.UUID, list[float]] = {}
    for _ in range(n_instruments):
        instr_id = uuid.uuid4()
        price = 100.0
        closes = [price]
        for _ in range(n_bars - 1):
            price *= 1 + rng.gauss(0.0005, 0.01)
            closes.append(max(price, 1.0))
        data[instr_id] = closes
    return data


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=_RUN_ID,
        strategy_name="test_cross_sectional",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _account(cash: Decimal = Decimal("500000")) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=cash,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=cash,
        net_asset_value=cash,
        positions=(),
    )


def _limits() -> RiskLimits:
    return RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=_RUN_ID,
        effective_from=_NOW,
        max_single_name_weight=Decimal("0.20"),
        max_sector_weight=Decimal("0.40"),
        max_gross_exposure=Decimal("0.95"),
        max_daily_turnover=Decimal("0.30"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.20"),
        vol_target_annualised=Decimal("0.15"),
    )


# ---------------------------------------------------------------------------
# Step 1+2: bar data → FeatureBundle
# ---------------------------------------------------------------------------


def test_feature_bundle_produced() -> None:
    bar_data = _make_bar_data(20, 300)
    bundle = build_feature_bundle(bar_data)
    assert len(bundle.alpha_features) > 0
    assert len(bundle.vol_forecasts) > 0


def test_feature_bundle_instruments_match() -> None:
    bar_data = _make_bar_data(15, 300)
    bundle = build_feature_bundle(bar_data)
    # Every instrument that appears in alpha_features should also have a vol forecast
    for instr_id in bundle.alpha_features:
        assert instr_id in bundle.vol_forecasts, (
            f"instrument {instr_id} has alpha features but no vol forecast"
        )


def test_feature_bundle_values_in_range() -> None:
    bar_data = _make_bar_data(10, 300)
    bundle = build_feature_bundle(bar_data)
    for instr_id, features in bundle.alpha_features.items():
        for name, value in features.items():
            assert -1.0 <= value <= 1.0, (
                f"feature {name} for {instr_id}: value {value} out of [-1,1]"
            )


def test_vol_forecasts_are_positive() -> None:
    bar_data = _make_bar_data(10, 300)
    bundle = build_feature_bundle(bar_data)
    for instr_id, vol in bundle.vol_forecasts.items():
        assert vol > 0.0, f"vol forecast for {instr_id} is non-positive: {vol}"


# ---------------------------------------------------------------------------
# Step 3: FeatureBundle → signal scores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_scores_produced() -> None:
    bar_data = _make_bar_data(20, 300)
    bundle = build_feature_bundle(bar_data)

    model = LinearWeightSignalModel(
        {
            "momentum_1m": 0.20,
            "momentum_3m": 0.30,
            "momentum_12m_1m": 0.40,
            "vol_compression": 0.10,
        }
    )
    controller = GenerateSignalsControllerImpl(model, _NullEventBus())
    run = _strategy_run()

    scores = await controller.generate(bundle.alpha_features, run, _NOW)

    assert len(scores) == len(bundle.alpha_features)
    assert all(-1.0 <= s.score <= 1.0 for s in scores)


@pytest.mark.asyncio
async def test_signal_scores_instrument_ids_match_bundle() -> None:
    bar_data = _make_bar_data(15, 300)
    bundle = build_feature_bundle(bar_data)

    model = LinearWeightSignalModel({"momentum_3m": 1.0})
    controller = GenerateSignalsControllerImpl(model, _NullEventBus())

    scores = await controller.generate(bundle.alpha_features, _strategy_run(), _NOW)
    scored_ids = {s.instrument_id for s in scores}
    assert scored_ids == set(bundle.alpha_features.keys())


@pytest.mark.asyncio
async def test_empty_bundle_returns_no_scores() -> None:
    model = LinearWeightSignalModel({"momentum_1m": 1.0})
    controller = GenerateSignalsControllerImpl(model, _NullEventBus())

    scores = await controller.generate({}, _strategy_run(), _NOW)
    assert scores == []


# ---------------------------------------------------------------------------
# Step 4: regime detection from bar data
# ---------------------------------------------------------------------------


def _make_index_closes(n: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    closes = [3000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0.0003, 0.008)))
    return closes


def test_compute_stats_returns_valid_market_stats() -> None:
    index_closes = _make_index_closes(220)
    bar_data = _make_bar_data(30, 60)
    instrument_closes = {k: v for k, v in bar_data.items()}
    stats = MarketRegimeDetector.compute_stats(index_closes, instrument_closes, _NOW)
    assert 0.0 <= stats.breadth <= 1.0
    assert stats.realized_vol > 0.0


def test_regime_detection_end_to_end() -> None:
    index_closes = _make_index_closes(220)
    bar_data = _make_bar_data(20, 60)
    instrument_closes = {k: v for k, v in bar_data.items()}

    detector = MarketRegimeDetector()
    stats = MarketRegimeDetector.compute_stats(index_closes, instrument_closes, _NOW)
    detector.update(stats)
    regime = detector.classify(stats)

    assert regime.regime_label in (
        RegimeLabel.RISK_ON,
        RegimeLabel.RISK_OFF,
        RegimeLabel.TRANSITION,
        RegimeLabel.CRISIS,
    )
    assert 0.0 <= regime.confidence <= 1.0


# ---------------------------------------------------------------------------
# Step 5: VolTargetedPortfolioConstructor end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_target_from_full_pipeline() -> None:
    """Full pipeline: bar data → bundle → signals → vol-targeted portfolio."""
    bar_data = _make_bar_data(20, 300)
    bundle = build_feature_bundle(bar_data)

    model = LinearWeightSignalModel(
        {
            "momentum_1m": 0.20,
            "momentum_3m": 0.30,
            "momentum_12m_1m": 0.40,
            "vol_compression": 0.10,
        }
    )
    controller = GenerateSignalsControllerImpl(model, _NullEventBus())
    signals = await controller.generate(bundle.alpha_features, _strategy_run(), _NOW)

    index_closes = _make_index_closes(220)
    stats = MarketRegimeDetector.compute_stats(
        index_closes, {k: v for k, v in bar_data.items()}, _NOW
    )
    detector = MarketRegimeDetector()
    detector.update(stats)
    regime = detector.classify(stats)

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=10)
    constructor.set_vol_forecasts(bundle.vol_forecasts)

    target = constructor.build_targets(signals, regime, _account(), _limits())

    # Basic invariants
    assert target.weights is not None
    total_invested = sum(target.weights.values())
    total = total_invested + target.cash_target_weight
    assert total <= Decimal("1.001")
    assert total >= Decimal("0.0")  # crisis regime may hold all cash


def test_portfolio_weights_respect_single_name_cap() -> None:
    bar_data = _make_bar_data(15, 300)
    bundle = build_feature_bundle(bar_data)

    # Give one instrument an artificially low vol to trigger the cap
    instr_ids = list(bundle.vol_forecasts.keys())
    boosted_forecasts = dict(bundle.vol_forecasts)
    boosted_forecasts[instr_ids[0]] = 0.01  # very low → very high raw weight

    signals = [_make_signal(instr_id, score=0.8) for instr_id in instr_ids]

    limits = _limits()
    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=15)
    constructor.set_vol_forecasts(boosted_forecasts)

    from quant_platform.core.domain.signals import RegimeState

    regime = RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=RegimeLabel.RISK_ON,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )
    target = constructor.build_targets(signals, regime, _account(), limits)

    for w in target.weights.values():
        assert w <= limits.max_single_name_weight + Decimal("0.001"), (
            f"weight {w} exceeds max_single_name_weight {limits.max_single_name_weight}"
        )


def _make_signal(instrument_id: uuid.UUID, score: float):
    from quant_platform.core.domain.signals import SignalScore

    return SignalScore(
        score_id=uuid.uuid4(),
        instrument_id=instrument_id,
        strategy_run_id=_RUN_ID,
        as_of=_NOW,
        score=score,
        confidence=1.0,
        model_version="test",
        feature_vector_id=None,
    )


# ---------------------------------------------------------------------------
# Regression: insufficient-data instruments are excluded from pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_series_excluded_from_pipeline() -> None:
    """Instruments with < 253 bars should be excluded from alpha features."""
    bar_data = _make_bar_data(10, 300)
    short_id = uuid.uuid4()
    bar_data[short_id] = [
        100.0 + i * 0.1 for i in range(10)
    ]  # only 10 bars — too few for any factor

    bundle = build_feature_bundle(bar_data)

    # short_id should not appear in alpha features
    assert short_id not in bundle.alpha_features

    # Running signals on the valid subset should still work
    model = LinearWeightSignalModel({"momentum_3m": 1.0})
    controller = GenerateSignalsControllerImpl(model, _NullEventBus())
    scores = await controller.generate(bundle.alpha_features, _strategy_run(), _NOW)

    scored_ids = {s.instrument_id for s in scores}
    assert short_id not in scored_ids
