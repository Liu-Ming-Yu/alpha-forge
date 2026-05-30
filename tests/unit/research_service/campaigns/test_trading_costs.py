"""Tests for the pluggable trading-cost layer (Arm K, ADR-007).

Covers four contracts:

1. **Behavior preservation** — ``evaluate_long_only_portfolio`` with
   ``cost_model=None`` is bit-identical to passing an explicit
   ``LinearTurnoverCost`` built from the same ``slippage_bps_per_turnover``.
   This is the guard that adding the cost seam did not change arms A-J.
2. **Linear cost fidelity** — ``LinearTurnoverCost`` charges exactly
   ``turnover * bps / 1e4``.
3. **Quadratic impact** — the convex term is per-name (concentration costs
   more than the same turnover spread across names), anchored at the documented
   calibration point, and always >= the linear-only cost.
4. **Driver integration** — ``run_sample_walk_forward`` accepts a cost model on
   the long-only path, rejects it on the signed-rank path, leaves the IC series
   untouched (cost does not change the alpha), and never raises returns.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.portfolio import (
    CampaignPortfolioConfig,
    LinearTurnoverCost,
    QuadraticImpactCost,
    TradingCostModel,
    evaluate_long_only_portfolio,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

# -- fixtures -------------------------------------------------------------------


def _rotating_scored(
    *, n_days: int = 12, n_instruments: int = 8
) -> list[tuple[SupervisedAlphaSample, float]]:
    """Scored rows whose ranking rotates daily, so the book actually trades.

    Realized returns are populated so the evaluator runs in realized mode. The
    score for each name oscillates by day + index so the top-N set churns,
    producing non-trivial turnover to price.
    """
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[tuple[SupervisedAlphaSample, float]] = []
    for d in range(n_days):
        as_of = start + timedelta(days=d)
        for i, inst in enumerate(insts):
            score = float((i + d) % n_instruments)
            sample = SupervisedAlphaSample(
                as_of=as_of,
                instrument_id=inst,
                features={"alpha": score},
                forward_return=0.01 if score > 0 else -0.005,
                realized_return_1d=0.001 * (1 if (i + d) % 2 == 0 else -1),
            )
            out.append((sample, score))
    return out


def _long_only_config() -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode="runtime-long-only",
        top_n=3,
        vol_target=0.15,
        vol_floor=0.05,
        vol_lookback_days=3,
        max_gross_exposure=0.5,
        min_cash_buffer=0.05,
        max_single_name_weight=0.2,
        max_daily_turnover=1.0,
        max_position_change=0.2,
        no_trade_band=0.0,
        rebalance_interval_days=1,
    )


# -- 1. behavior preservation ---------------------------------------------------


class TestDefaultMatchesExplicitLinear:
    @pytest.mark.parametrize("bps", [0.0, 10.0, 37.5])
    def test_none_equals_linear(self, bps: float) -> None:
        scored = _rotating_scored()
        config = _long_only_config()
        default = evaluate_long_only_portfolio(scored, slippage_bps_per_turnover=bps, config=config)
        explicit = evaluate_long_only_portfolio(
            scored,
            slippage_bps_per_turnover=bps,
            config=config,
            cost_model=LinearTurnoverCost(bps),
        )
        assert default.daily_returns == explicit.daily_returns
        assert default.daily_turnover == explicit.daily_turnover
        assert default.daily_ics == explicit.daily_ics


# -- 2. linear cost fidelity ----------------------------------------------------


class TestLinearTurnoverCost:
    def test_cost_is_turnover_times_bps(self) -> None:
        model = LinearTurnoverCost(10.0)
        a, b = uuid.uuid4(), uuid.uuid4()
        # turnover = |0.1| + |-0.05| = 0.15; cost = 0.15 * 10 / 1e4
        assert model.cost({a: 0.1, b: -0.05}) == pytest.approx(0.15 * 10.0 / 1e4)

    def test_empty_and_zero_trades_cost_nothing(self) -> None:
        model = LinearTurnoverCost(10.0)
        assert model.cost({}) == 0.0
        assert model.cost({uuid.uuid4(): 0.0}) == 0.0

    def test_name_encodes_bps(self) -> None:
        assert LinearTurnoverCost(10.0).name == "linear-10bps"
        assert LinearTurnoverCost(0.0).name == "linear-0bps"

    def test_rejects_negative_bps(self) -> None:
        with pytest.raises(ValueError, match="bps_per_turnover must be >= 0"):
            LinearTurnoverCost(-1.0)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LinearTurnoverCost(10.0), TradingCostModel)


# -- 3. quadratic impact --------------------------------------------------------


class TestQuadraticImpactCost:
    def test_linear_plus_quadratic_decomposition(self) -> None:
        # linear_bps=10, impact 10 bps at cap 0.05 -> quad_coef = 0.001/0.0025 = 0.4
        model = QuadraticImpactCost(
            linear_bps_per_turnover=10.0, impact_bps_at_cap=10.0, single_name_cap=0.05
        )
        a = uuid.uuid4()
        # one name traded at the cap: linear = 0.05*10/1e4; quad = 0.4*0.05**2 = 0.001
        expected = 0.05 * 10.0 / 1e4 + 0.4 * 0.05**2
        assert model.cost({a: 0.05}) == pytest.approx(expected)

    def test_anchor_impact_equals_impact_bps_at_cap(self) -> None:
        # With no linear term, a single cap-sized trade costs exactly the anchor.
        model = QuadraticImpactCost(
            linear_bps_per_turnover=0.0, impact_bps_at_cap=10.0, single_name_cap=0.05
        )
        assert model.cost({uuid.uuid4(): 0.05}) == pytest.approx(10.0 / 1e4)

    def test_concentration_costs_more_than_spread(self) -> None:
        # Same total turnover (0.10), concentrated in one name vs split across two.
        model = QuadraticImpactCost(linear_bps_per_turnover=10.0)
        a, b = uuid.uuid4(), uuid.uuid4()
        concentrated = model.cost({a: 0.10})
        spread = model.cost({a: 0.05, b: 0.05})
        assert concentrated > spread
        # Linear components are equal (same turnover); only the quadratic differs:
        # 0.10**2 = 0.01 vs 0.05**2 + 0.05**2 = 0.005.
        quad_coef = (10.0 / 1e4) / 0.05**2
        assert concentrated - spread == pytest.approx(quad_coef * (0.01 - 0.005))

    def test_always_at_least_linear_only(self) -> None:
        quad = QuadraticImpactCost(linear_bps_per_turnover=10.0, impact_bps_at_cap=10.0)
        linear = LinearTurnoverCost(10.0)
        a, b = uuid.uuid4(), uuid.uuid4()
        trades = {a: 0.07, b: -0.03}
        assert quad.cost(trades) >= linear.cost(trades)

    def test_empty_trades_cost_nothing(self) -> None:
        assert QuadraticImpactCost().cost({}) == 0.0

    def test_metadata_round_trips_params(self) -> None:
        meta = QuadraticImpactCost(
            linear_bps_per_turnover=10.0, impact_bps_at_cap=12.0, single_name_cap=0.04
        ).metadata()
        assert meta["type"] == "quadratic_impact"
        assert meta["linear_bps_per_turnover"] == 10.0
        assert meta["impact_bps_at_cap"] == 12.0
        assert meta["single_name_cap"] == 0.04
        assert meta["quad_coef"] == pytest.approx((12.0 / 1e4) / 0.04**2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"linear_bps_per_turnover": -1.0}, "linear_bps_per_turnover must be >= 0"),
            ({"impact_bps_at_cap": -1.0}, "impact_bps_at_cap must be >= 0"),
            ({"single_name_cap": 0.0}, "single_name_cap must be in"),
            ({"single_name_cap": 1.5}, "single_name_cap must be in"),
        ],
    )
    def test_rejects_bad_params(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            QuadraticImpactCost(**kwargs)  # type: ignore[arg-type]

    def test_satisfies_protocol(self) -> None:
        assert isinstance(QuadraticImpactCost(), TradingCostModel)


# -- 4. driver integration ------------------------------------------------------


def _wf_samples() -> list[SupervisedAlphaSample]:
    return [sample for sample, _ in _rotating_scored(n_days=40, n_instruments=8)]


def _wf_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=10,
        test_window_days=3,
        step_days=3,
        purge_days=1,
        embargo_days=0,
        min_folds=1,
        label_horizon_days=1,
    )


class TestWalkForwardCostModel:
    def test_quadratic_leaves_ics_and_lowers_returns(self) -> None:
        kwargs = dict(
            samples=_wf_samples(),
            config=_wf_config(),
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["alpha"],
            slippage_bps_per_turnover=10.0,
            portfolio_config=_long_only_config(),
        )
        baseline = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        quad = run_sample_walk_forward(
            cost_model=QuadraticImpactCost(linear_bps_per_turnover=10.0, impact_bps_at_cap=10.0),
            **kwargs,  # type: ignore[arg-type]
        )
        # Cost never touches the ranking, so the IC series is identical...
        assert quad.daily_ics == baseline.daily_ics
        # ...but the convex cost is >= the linear-only cost, so every daily return
        # is <= the baseline's (costs only subtract).
        assert all(q <= b + 1e-12 for q, b in zip(quad.daily_returns, baseline.daily_returns))
        assert quad.metrics["total_return"] <= baseline.metrics["total_return"] + 1e-12

    def test_cost_model_metadata_pinned_only_when_supplied(self) -> None:
        kwargs = dict(
            samples=_wf_samples(),
            config=_wf_config(),
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["alpha"],
            portfolio_config=_long_only_config(),
        )
        baseline = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        quad = run_sample_walk_forward(
            cost_model=QuadraticImpactCost(),
            **kwargs,  # type: ignore[arg-type]
        )
        assert "cost_model" not in baseline.portfolio_diagnostics
        assert quad.portfolio_diagnostics["cost_model"]["type"] == "quadratic_impact"  # type: ignore[index]

    def test_cost_model_rejected_without_portfolio(self) -> None:
        with pytest.raises(ValueError, match="cost_model requires a portfolio_config"):
            run_sample_walk_forward(
                samples=_wf_samples(),
                config=_wf_config(),
                model_version="test",
                feature_set_version="test-fs",
                feature_names=["alpha"],
                cost_model=QuadraticImpactCost(),
            )
