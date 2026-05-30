"""Live LongOnlyPortfolioConstructor no-trade band — WS5 research/production parity."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.algorithms.portfolio_construction import LongOnlyPortfolioConstructor
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot
from quant_platform.core.domain.signals import RegimeLabel, RegimeState, SignalScore

_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=UTC)
_RUN_ID = uuid.uuid4()


def _regime() -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=RegimeLabel.RISK_ON,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
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


def _position(instrument_id: uuid.UUID, market_value: str) -> PositionSnapshot:
    return PositionSnapshot(
        snapshot_id=uuid.uuid4(),
        instrument_id=instrument_id,
        quantity=1,
        average_cost=Decimal(market_value),
        market_price=Decimal(market_value),
        market_value=Decimal(market_value),
        unrealised_pnl=Decimal("0"),
        as_of=_NOW,
    )


def _account(positions: tuple[PositionSnapshot, ...] = ()) -> AccountSnapshot:
    cash = Decimal("100000")
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=cash,
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=cash,
        net_asset_value=cash,
        positions=positions,
    )


def test_no_trade_band_holds_position_within_band() -> None:
    """A held name whose target is within the band keeps its current weight."""
    ids = [uuid.uuid4() for _ in range(10)]
    signals = [_signal(i, 0.5) for i in ids]
    # NAV 100000; instrument ids[0] held at market value 9400 -> current weight 0.094.
    account = _account((_position(ids[0], "9400"),))

    constructor = LongOnlyPortfolioConstructor(top_n=10, no_trade_band=0.01)
    target = constructor.build_targets(signals, _regime(), account, _limits())

    # Equal-weight target is 0.95 / 10 = 0.095; |0.095 - 0.094| = 0.001 < band.
    assert target.weights[ids[0]] == Decimal("0.094")
    assert target.weights[ids[1]] == Decimal("0.095")


def test_no_trade_band_zero_is_noop() -> None:
    ids = [uuid.uuid4() for _ in range(10)]
    signals = [_signal(i, 0.5) for i in ids]
    account = _account((_position(ids[0], "9400"),))

    constructor = LongOnlyPortfolioConstructor(top_n=10, no_trade_band=0.0)
    target = constructor.build_targets(signals, _regime(), account, _limits())

    assert all(w == Decimal("0.095") for w in target.weights.values())


def test_no_trade_band_carries_sub_band_unselected_position() -> None:
    """A held name that is no longer selected is carried if below the band."""
    ids = [uuid.uuid4() for _ in range(10)]
    signals = [_signal(i, 0.5) for i in ids]
    stale = uuid.uuid4()  # not in signals -> target weight 0
    # stale held at market value 500 -> current weight 0.005 < band 0.01.
    account = _account((_position(stale, "500"),))

    constructor = LongOnlyPortfolioConstructor(top_n=10, no_trade_band=0.01)
    target = constructor.build_targets(signals, _regime(), account, _limits())

    assert target.weights[stale] == Decimal("0.005")
    assert any("no-trade band carried" in note for note in target.construction_notes)


def test_negative_no_trade_band_raises() -> None:
    with pytest.raises(ValueError, match="no_trade_band must be >= 0"):
        LongOnlyPortfolioConstructor(top_n=10, no_trade_band=-0.01)


# -- conviction weighting (Arm Q live deploy; ADR/construction-cost framework) ---


def test_conviction_mode_tilts_toward_higher_scores() -> None:
    """conviction_shrinkage set ⇒ the top-N are sized by conviction (highest
    score gets the most weight), unlike the equal-weight default. The selection
    is unchanged; only sizing differs. Weights stay long-only, within the
    per-name cap, and inside the investable-gross budget."""
    ids = [uuid.uuid4() for _ in range(5)]
    scores = [0.75, 0.70, 0.66, 0.62, 0.60]
    signals = [_signal(i, s) for i, s in zip(ids, scores, strict=True)]

    conviction = LongOnlyPortfolioConstructor(top_n=5, conviction_shrinkage=0.25)
    target = conviction.build_targets(signals, _regime(), _account(), _limits())

    weights = target.weights
    assert set(weights) == set(ids)  # selection unchanged — all five held
    # Conviction tilt: strictly decreasing weight with decreasing score.
    ordered = [float(weights[i]) for i in ids]
    assert ordered == sorted(ordered, reverse=True)
    assert ordered[0] > ordered[-1]
    assert all(w > 0 for w in ordered)
    # Long-only within budget: gross <= investable, each <= per-name cap.
    limits = _limits()
    investable = float(min(limits.max_gross_exposure, Decimal("1") - limits.min_cash_buffer))
    regime_scale = 1.0  # RISK_ON
    assert sum(ordered) <= investable * regime_scale + 1e-9
    assert all(w <= float(limits.max_single_name_weight) + 1e-9 for w in ordered)


def test_equal_weight_default_unchanged_by_conviction_param() -> None:
    """conviction_shrinkage=None (default) ⇒ exact equal weight — behaviour
    preserved for every existing strategy."""
    ids = [uuid.uuid4() for _ in range(4)]
    signals = [_signal(i, s) for i, s in zip(ids, [0.9, 0.7, 0.6, 0.55], strict=True)]
    equal = LongOnlyPortfolioConstructor(top_n=4)  # default: no conviction
    target = equal.build_targets(signals, _regime(), _account(), _limits())
    vals = [float(w) for w in target.weights.values()]
    assert vals  # non-empty
    assert all(abs(v - vals[0]) < 1e-12 for v in vals)  # all identical (equal weight)
