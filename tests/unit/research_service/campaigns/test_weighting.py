"""Tests for the pluggable position-weighting layer (Arm L, ADR-008).

Covers four contracts:

1. **Behavior preservation** — ``raw_long_only_target`` with ``weighting=None``
   is bit-identical to an explicit ``EqualWeight`` and to the prior
   equal-weight arithmetic. This is the guard that adding the seam did not
   change arms A-K.
2. **Equal-weight fidelity** — ``EqualWeight`` returns ``1/N`` per name.
3. **Inverse-vol weighting** — proportions sum to 1, lower vol earns more
   weight, ``shrinkage`` interpolates equal↔inverse-vol, missing vol falls back
   to the cross-sectional median, and parameters validate.
4. **Driver integration** — ``run_sample_walk_forward`` accepts a weighting
   scheme on the long-only path, rejects it on the signed-rank path, leaves the
   IC series and selected set untouched (weighting does not change the alpha),
   and pins its metadata only when supplied.
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
    EqualWeight,
    InverseVolWeight,
    WeightingScheme,
)
from quant_platform.services.research_service.campaigns.portfolio.targets import (
    raw_long_only_target,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

# -- fixtures -------------------------------------------------------------------


def _row(score: float, *, vol: float | None = None) -> tuple[SupervisedAlphaSample, float]:
    features: dict[str, float] = {"alpha": score}
    if vol is not None:
        # The price-volume family stores sign-flipped std, so low_vol = -vol.
        features["low_vol_63d"] = -vol
    sample = SupervisedAlphaSample(
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        instrument_id=uuid.uuid4(),
        features=features,
        forward_return=0.0,
    )
    return sample, score


def _config(top_n: int = 4) -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode="runtime-long-only",
        top_n=top_n,
        max_gross_exposure=0.22,
        min_cash_buffer=0.05,
        max_single_name_weight=0.05,
    )


# -- 1. behavior preservation ---------------------------------------------------


class TestRawTargetBehaviorPreservation:
    def test_none_equals_explicit_equal_weight(self) -> None:
        rows = [_row(3.0), _row(2.0), _row(1.0), _row(-1.0)]
        cfg = _config()
        default = raw_long_only_target(rows, config=cfg)
        explicit = raw_long_only_target(rows, config=cfg, weighting=EqualWeight())
        assert default == explicit

    def test_matches_prior_equal_weight_formula(self) -> None:
        rows = [_row(3.0), _row(2.0), _row(1.0)]  # 3 positive names
        cfg = _config()
        target = raw_long_only_target(rows, config=cfg)
        investable = min(cfg.max_gross_exposure, 1.0 - cfg.min_cash_buffer)
        expected_per_name = min(cfg.max_single_name_weight, investable / 3)
        assert all(w == pytest.approx(expected_per_name) for w in target.values())

    def test_shrinkage_one_equals_equal_weight(self) -> None:
        rows = [_row(3.0, vol=0.01), _row(2.0, vol=0.03), _row(1.0, vol=0.02)]
        cfg = _config()
        equal = raw_long_only_target(rows, config=cfg, weighting=EqualWeight())
        shrunk = raw_long_only_target(rows, config=cfg, weighting=InverseVolWeight(shrinkage=1.0))
        assert shrunk.keys() == equal.keys()
        for key in equal:
            assert shrunk[key] == pytest.approx(equal[key])


# -- 2. equal-weight fidelity ---------------------------------------------------


class TestEqualWeight:
    def test_uniform_proportions(self) -> None:
        rows = [_row(1.0), _row(2.0), _row(3.0)]
        assert EqualWeight().proportions(rows) == [pytest.approx(1 / 3)] * 3

    def test_empty(self) -> None:
        assert EqualWeight().proportions([]) == []

    def test_satisfies_protocol(self) -> None:
        assert isinstance(EqualWeight(), WeightingScheme)


# -- 3. inverse-vol weighting ---------------------------------------------------


class TestInverseVolWeight:
    def test_proportions_sum_to_one(self) -> None:
        rows = [_row(1.0, vol=0.01), _row(2.0, vol=0.02), _row(3.0, vol=0.04)]
        props = InverseVolWeight(shrinkage=0.0).proportions(rows)
        assert sum(props) == pytest.approx(1.0)
        assert all(p >= 0 for p in props)

    def test_lower_vol_gets_more_weight(self) -> None:
        # Pure inverse-vol: the lowest-vol name must get the largest proportion.
        rows = [_row(1.0, vol=0.01), _row(2.0, vol=0.02), _row(3.0, vol=0.04)]
        props = InverseVolWeight(shrinkage=0.0).proportions(rows)
        assert props[0] > props[1] > props[2]

    def test_shrinkage_interpolates(self) -> None:
        rows = [_row(1.0, vol=0.01), _row(2.0, vol=0.04)]
        pure = InverseVolWeight(shrinkage=0.0).proportions(rows)
        half = InverseVolWeight(shrinkage=0.5).proportions(rows)
        # Half-shrunk weights sit strictly between pure inverse-vol and 1/N.
        assert pure[0] > half[0] > 0.5
        assert pure[1] < half[1] < 0.5

    def test_missing_vol_falls_back_to_median(self) -> None:
        # Middle name has no vol feature; it should not crash and weights still
        # sum to 1 (the missing name is imputed at the median of the valid set).
        rows = [_row(1.0, vol=0.01), _row(2.0), _row(3.0, vol=0.04)]
        props = InverseVolWeight(shrinkage=0.0).proportions(rows)
        assert sum(props) == pytest.approx(1.0)
        assert len(props) == 3

    def test_all_missing_vol_does_not_crash(self) -> None:
        rows = [_row(1.0), _row(2.0)]
        props = InverseVolWeight(shrinkage=0.0).proportions(rows)
        assert sum(props) == pytest.approx(1.0)

    def test_vol_floor_caps_tiny_vol(self) -> None:
        # A near-zero-vol name would dominate without the floor; with it, its
        # weight is bounded relative to a name at the floor.
        rows = [_row(1.0, vol=1e-9), _row(2.0, vol=0.005)]
        props = InverseVolWeight(shrinkage=0.0, vol_floor=0.005).proportions(rows)
        assert props[0] == pytest.approx(props[1])  # both clamped to the floor

    def test_empty(self) -> None:
        assert InverseVolWeight().proportions([]) == []

    def test_metadata_round_trips(self) -> None:
        meta = InverseVolWeight(vol_feature="low_vol_63d", shrinkage=0.5, vol_floor=0.01).metadata()
        assert meta["type"] == "inverse_vol"
        assert meta["vol_feature"] == "low_vol_63d"
        assert meta["shrinkage"] == 0.5
        assert meta["vol_floor"] == 0.01

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"shrinkage": -0.1}, "shrinkage must be in"),
            ({"shrinkage": 1.1}, "shrinkage must be in"),
            ({"vol_floor": 0.0}, "vol_floor must be > 0"),
            ({"vol_feature": ""}, "vol_feature must be"),
        ],
    )
    def test_rejects_bad_params(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            InverseVolWeight(**kwargs)  # type: ignore[arg-type]

    def test_satisfies_protocol(self) -> None:
        assert isinstance(InverseVolWeight(), WeightingScheme)


# -- 4. driver integration ------------------------------------------------------


def _wf_samples(*, n_days: int = 40, n_instruments: int = 8) -> list[SupervisedAlphaSample]:
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[SupervisedAlphaSample] = []
    for d in range(n_days):
        as_of = start + timedelta(days=d)
        for i, inst in enumerate(insts):
            score = float((i + d) % n_instruments)
            out.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=inst,
                    features={"alpha": score, "low_vol_63d": -0.01 * (1 + i)},
                    forward_return=0.01 if score > 0 else -0.005,
                    realized_return_1d=0.001 * (1 if (i + d) % 2 == 0 else -1),
                )
            )
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


class TestWalkForwardWeighting:
    def test_inverse_vol_leaves_ics_changes_returns(self) -> None:
        kwargs = dict(
            samples=_wf_samples(),
            config=_wf_config(),
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["alpha"],
            portfolio_config=_long_only_config(),
        )
        baseline = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        invvol = run_sample_walk_forward(
            weighting=InverseVolWeight(shrinkage=0.5),
            **kwargs,  # type: ignore[arg-type]
        )
        # Weighting never touches selection or scoring, so the IC series is
        # identical; only the sizing (and thus returns) differs.
        assert invvol.daily_ics == baseline.daily_ics
        assert invvol.daily_returns != baseline.daily_returns

    def test_weighting_metadata_pinned_only_when_supplied(self) -> None:
        kwargs = dict(
            samples=_wf_samples(),
            config=_wf_config(),
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["alpha"],
            portfolio_config=_long_only_config(),
        )
        baseline = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        invvol = run_sample_walk_forward(
            weighting=InverseVolWeight(),
            **kwargs,  # type: ignore[arg-type]
        )
        assert "weighting" not in baseline.portfolio_diagnostics
        assert invvol.portfolio_diagnostics["weighting"]["type"] == "inverse_vol"  # type: ignore[index]

    def test_weighting_rejected_without_portfolio(self) -> None:
        with pytest.raises(ValueError, match="weighting requires a portfolio_config"):
            run_sample_walk_forward(
                samples=_wf_samples(),
                config=_wf_config(),
                model_version="test",
                feature_set_version="test-fs",
                feature_names=["alpha"],
                weighting=InverseVolWeight(),
            )
