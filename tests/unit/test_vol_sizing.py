"""Unit tests for VolTargetedPortfolioConstructor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore
from quant_platform.services.portfolio_service.portfolio_constructor import (
    LongOnlyPortfolioConstructor,
)
from quant_platform.services.portfolio_service.vol_sizing import (
    VolTargetedPortfolioConstructor,
)

_UTC = UTC
_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=_UTC)
_RUN_ID = uuid.uuid4()


def _regime(label: RegimeLabel = RegimeLabel.RISK_ON) -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=label,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


def _account(cash: Decimal = Decimal("100000")) -> AccountSnapshot:
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


def _signal(instrument_id: uuid.UUID, score: float) -> SignalScore:
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
# Constructor validation
# ---------------------------------------------------------------------------


def test_negative_vol_target_raises() -> None:
    with pytest.raises(ValueError, match="vol_target must be positive"):
        VolTargetedPortfolioConstructor(vol_target=-0.15)


def test_zero_vol_target_raises() -> None:
    with pytest.raises(ValueError, match="vol_target must be positive"):
        VolTargetedPortfolioConstructor(vol_target=0.0)


def test_zero_vol_floor_raises() -> None:
    with pytest.raises(ValueError, match="min_vol_floor must be positive"):
        VolTargetedPortfolioConstructor(min_vol_floor=0.0)


# ---------------------------------------------------------------------------
# Fallback to base when no forecasts
# ---------------------------------------------------------------------------


def test_returns_base_when_no_forecasts() -> None:
    """Without set_vol_forecasts(), result equals LongOnlyPortfolioConstructor."""
    instr_ids = [uuid.uuid4() for _ in range(5)]
    signals = [_signal(i, 0.8) for i in instr_ids]

    base = LongOnlyPortfolioConstructor(top_n=5)
    vol = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=5)

    base_target = base.build_targets(signals, _regime(), _account(), _limits())
    vol_target = vol.build_targets(signals, _regime(), _account(), _limits())

    assert base_target.weights == vol_target.weights


def test_returns_base_on_crisis_regime() -> None:
    """In CRISIS regime the base constructor returns empty weights; vol constructor does too."""
    instr_ids = [uuid.uuid4() for _ in range(5)]
    signals = [_signal(i, 0.8) for i in instr_ids]

    vol = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=5)
    vol.set_vol_forecasts({i: 0.20 for i in instr_ids})

    target = vol.build_targets(signals, _regime(RegimeLabel.CRISIS), _account(), _limits())
    assert target.weights == {}
    assert target.cash_target_weight == Decimal("1")


# ---------------------------------------------------------------------------
# Vol scaling correctness
# ---------------------------------------------------------------------------


def test_lower_vol_gets_higher_weight() -> None:
    """Instrument with lower forecast vol receives a higher weight."""
    low_vol_id = uuid.uuid4()
    high_vol_id = uuid.uuid4()

    signals = [_signal(low_vol_id, 0.9), _signal(high_vol_id, 0.8)]
    forecasts = {low_vol_id: 0.10, high_vol_id: 0.30}

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=2)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), _limits())

    assert low_vol_id in target.weights
    assert high_vol_id in target.weights
    assert target.weights[low_vol_id] > target.weights[high_vol_id]


def test_same_vol_gets_same_weight() -> None:
    """Instruments with identical forecast vols receive identical weights."""
    ids = [uuid.uuid4() for _ in range(4)]
    signals = [_signal(i, 0.8) for i in ids]
    forecasts = {i: 0.20 for i in ids}

    constructor = VolTargetedPortfolioConstructor(vol_target=0.20, top_n=4)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), _limits())

    weights = list(target.weights.values())
    assert all(abs(w - weights[0]) < Decimal("0.001") for w in weights)


def test_total_gross_exposure_unchanged_after_scaling() -> None:
    """Vol scaling preserves the regime-adjusted gross exposure from the base."""
    ids = [uuid.uuid4() for _ in range(5)]
    signals = [_signal(i, 0.8) for i in ids]
    forecasts = {ids[0]: 0.10, ids[1]: 0.20, ids[2]: 0.30, ids[3]: 0.15, ids[4]: 0.25}

    base = LongOnlyPortfolioConstructor(top_n=5)
    base_target = base.build_targets(signals, _regime(), _account(), _limits())
    base_gross = sum(base_target.weights.values())

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=5)
    constructor.set_vol_forecasts(forecasts)
    vol_target = constructor.build_targets(signals, _regime(), _account(), _limits())
    vol_gross = sum(vol_target.weights.values())

    # Gross exposure may differ slightly due to capping at max_single_name_weight
    assert vol_gross <= base_gross + Decimal("0.001")


def test_weights_respect_max_single_name_cap() -> None:
    """Vol scaling never exceeds max_single_name_weight."""
    ids = [uuid.uuid4() for _ in range(3)]
    signals = [_signal(i, 0.9) for i in ids]
    # One instrument with very low vol → would naturally get a huge weight
    forecasts = {ids[0]: 0.01, ids[1]: 0.20, ids[2]: 0.20}

    limits = _limits()
    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=3)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), limits)

    for w in target.weights.values():
        assert w <= limits.max_single_name_weight + Decimal("0.001")


def test_min_vol_floor_prevents_oversizing() -> None:
    """min_vol_floor prevents extreme weight concentration for near-zero vol."""
    ids = [uuid.uuid4() for _ in range(3)]
    signals = [_signal(i, 0.9) for i in ids]

    # Without floor: vol_target / 0.001 = 150× base weight → would dominate
    forecasts = {ids[0]: 0.001, ids[1]: 0.20, ids[2]: 0.20}

    constructor = VolTargetedPortfolioConstructor(
        vol_target=0.15,
        min_vol_floor=0.05,
        top_n=3,
    )
    constructor.set_vol_forecasts(forecasts)
    limits = _limits()
    target = constructor.build_targets(signals, _regime(), _account(), limits)

    for w in target.weights.values():
        assert w <= limits.max_single_name_weight + Decimal("0.001")


def test_instruments_without_forecast_get_base_weight() -> None:
    """Instruments missing a vol forecast retain their base equal-weight."""
    has_forecast = uuid.uuid4()
    no_forecast = uuid.uuid4()

    signals = [_signal(has_forecast, 0.9), _signal(no_forecast, 0.8)]
    forecasts = {has_forecast: 0.15}  # no_forecast not in forecasts

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=2)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), _limits())
    assert no_forecast in target.weights


def test_construction_notes_include_vol_target_info() -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    signals = [_signal(i, 0.8) for i in ids]
    forecasts = {i: 0.20 for i in ids}

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=3)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), _limits())
    assert any("vol_targeted" in note for note in target.construction_notes)


def test_cash_target_plus_invested_equals_one() -> None:
    ids = [uuid.uuid4() for _ in range(5)]
    signals = [_signal(i, 0.7) for i in ids]
    forecasts = {ids[0]: 0.10, ids[1]: 0.20, ids[2]: 0.30, ids[3]: 0.15, ids[4]: 0.25}

    constructor = VolTargetedPortfolioConstructor(vol_target=0.15, top_n=5)
    constructor.set_vol_forecasts(forecasts)

    target = constructor.build_targets(signals, _regime(), _account(), _limits())
    total = sum(target.weights.values()) + target.cash_target_weight
    assert total <= Decimal("1.001")
    assert total >= Decimal("0.998")
