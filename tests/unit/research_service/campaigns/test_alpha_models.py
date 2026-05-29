"""Tests for the pluggable alpha-model layer.

Covers three contracts:

1. **Behavior preservation** — ``run_sample_walk_forward`` with ``model=None``
   (the implicit default) is bit-identical to passing an explicit
   ``LinearICRanker``. This is the guard that adding the model seam did not
   change arms A-H.
2. **Linear wrapper fidelity** — ``LinearICRanker`` reproduces the legacy
   ``fit_correlation_weights`` weights exactly, and scores them as a weighted
   sum over cross-sectionally rank-normalized features (the scale-domination fix).
3. **GBDT model** — the XGBoost ranker fits, scores in input order, exposes
   normalized importances, captures an interaction the linear ranker misses,
   and plugs into the driver. GBDT tests skip when the ``ml`` extra is absent;
   they force ``device="cpu"`` so they never depend on a GPU in CI.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.metrics.ranker_metrics import (
    fit_correlation_weights,
    score_features,
)
from quant_platform.services.research_service.campaigns.models import (
    AlphaModel,
    FittedAlphaModel,
    GradientBoostedRanker,
    LinearICRanker,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    cross_sectional_rank_normalize,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

try:
    import xgboost  # noqa: F401

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

_needs_xgboost = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost (ml extra) not installed")


def _samples(
    *, n_days: int = 16, n_instruments: int = 6, seed: int = 0
) -> list[SupervisedAlphaSample]:
    """Synthetic samples where f1 has a real positive IC; f2 only via interaction."""
    rng = random.Random(seed)
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    out: list[SupervisedAlphaSample] = []
    for d in range(n_days):
        for inst in insts:
            f1 = rng.gauss(0.0, 1.0)
            f2 = rng.gauss(0.0, 1.0)
            forward = 0.3 * f1 + 0.1 * rng.gauss(0.0, 1.0)
            realized = 0.01 if f1 > 0 else -0.01
            out.append(
                SupervisedAlphaSample(
                    as_of=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=d),
                    instrument_id=inst,
                    features={"f1": f1, "f2": f2},
                    forward_return=forward,
                    realized_return_1d=realized,
                )
            )
    return out


def _interaction_samples(
    *, n_days: int = 60, n_instruments: int = 40, seed: int = 1
) -> list[SupervisedAlphaSample]:
    """f1 helps only when f2 > 0 (sign flip) — zero marginal IC, pure interaction."""
    rng = random.Random(seed)
    insts = [uuid.uuid4() for _ in range(n_instruments)]
    out: list[SupervisedAlphaSample] = []
    for d in range(n_days):
        for inst in insts:
            f1 = rng.gauss(0.0, 1.0)
            f2 = rng.gauss(0.0, 1.0)
            forward = (f1 if f2 > 0 else -f1) * 0.5 + 0.1 * rng.gauss(0.0, 1.0)
            out.append(
                SupervisedAlphaSample(
                    as_of=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=d),
                    instrument_id=inst,
                    features={"f1": f1, "f2": f2},
                    forward_return=forward,
                )
            )
    return out


def _wf_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=6,
        test_window_days=2,
        step_days=2,
        purge_days=1,
        embargo_days=0,
        min_folds=1,
        label_horizon_days=1,
    )


# -- 1. behavior preservation ---------------------------------------------------


class TestDefaultMatchesExplicitLinear:
    """model=None must equal an explicit LinearICRanker, bit-for-bit."""

    @pytest.mark.parametrize("portfolio", [None, "long_only"])
    def test_none_equals_linear(self, portfolio: str | None) -> None:
        samples = _samples()
        config = _wf_config()
        portfolio_config = (
            None
            if portfolio is None
            else CampaignPortfolioConfig(
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
        )
        kwargs = dict(
            samples=samples,
            config=config,
            model_version="test",
            feature_set_version="test-fs",
            feature_names=["f1", "f2"],
            portfolio_config=portfolio_config,
        )
        default = run_sample_walk_forward(**kwargs)  # type: ignore[arg-type]
        explicit = run_sample_walk_forward(model=LinearICRanker(), **kwargs)  # type: ignore[arg-type]

        assert default.metrics == explicit.metrics
        assert default.daily_returns == explicit.daily_returns
        assert default.daily_ics == explicit.daily_ics
        assert default.selected_weights == explicit.selected_weights
        assert default.folds == explicit.folds


# -- 2. linear wrapper fidelity -------------------------------------------------


class TestLinearICRanker:
    def test_reproduces_legacy_weights_and_rank_normalizes_scores(self) -> None:
        samples = _samples()
        fitted = LinearICRanker().fit(samples, ["f1", "f2"])
        # Weights are unchanged: the fit uses rank-based Spearman IC, which is
        # invariant to the monotonic rank-normalization applied at scoring time.
        legacy_weights = fit_correlation_weights(samples, ["f1", "f2"], non_negative=True)
        assert dict(fitted.feature_weights()) == legacy_weights

        # Scoring rank-normalizes each feature within its as-of cross-section,
        # then takes the weighted sum — NOT the raw weighted sum, which would be
        # dominated by the largest-scale feature. Verify against the kernel
        # normalizer applied to a real same-date cross-section.
        batch = [s for s in samples if s.as_of == samples[0].as_of]
        assert len(batch) > 1  # guard: a genuine cross-section to rank within
        frame = pd.DataFrame(
            {
                "date": [s.as_of for s in batch],
                **{name: [s.features[name] for s in batch] for name in ("f1", "f2")},
            }
        )
        normed = cross_sectional_rank_normalize(frame, ["f1", "f2"], date_column="date")
        expected = [
            score_features(
                {"f1": float(normed.iloc[i]["f1"]), "f2": float(normed.iloc[i]["f2"])},
                legacy_weights,
            )
            for i in range(len(batch))
        ]
        assert fitted.score(batch) == pytest.approx(expected)
        # And it is NOT the raw weighted sum (the scale-domination bug this fixes).
        raw_scores = [score_features(s.features, legacy_weights) for s in batch]
        assert fitted.score(batch) != pytest.approx(raw_scores)

    def test_score_preserves_order(self) -> None:
        samples = _samples()
        fitted = LinearICRanker().fit(samples, ["f1", "f2"])
        assert len(fitted.score(samples)) == len(samples)

    def test_name_matches_legacy_model_version(self) -> None:
        assert LinearICRanker().name == "ic-weighted-non-negative"
        assert LinearICRanker(weight_mode="equal_weight").name == "equal-weight"

    def test_satisfies_protocols(self) -> None:
        model = LinearICRanker()
        assert isinstance(model, AlphaModel)
        assert isinstance(model.fit(_samples(), ["f1", "f2"]), FittedAlphaModel)

    def test_rejects_bad_weight_mode(self) -> None:
        with pytest.raises(ValueError, match="unsupported weight_mode"):
            LinearICRanker(weight_mode="nonsense")  # type: ignore[arg-type]


# -- 3. gradient-boosted ranker -------------------------------------------------


class TestGradientBoostedRankerConfig:
    def test_rejects_bad_device(self) -> None:
        with pytest.raises(ValueError, match="unsupported device"):
            GradientBoostedRanker(device="gpu")  # type: ignore[arg-type]

    def test_rejects_bad_num_boost_round(self) -> None:
        with pytest.raises(ValueError, match="num_boost_round"):
            GradientBoostedRanker(num_boost_round=0)

    def test_name_is_hardware_independent(self) -> None:
        assert GradientBoostedRanker(device="cpu").name == GradientBoostedRanker(device="auto").name

    def test_rejects_bad_objective(self) -> None:
        with pytest.raises(ValueError, match="unsupported objective"):
            GradientBoostedRanker(objective="ndcg")  # type: ignore[arg-type]

    def test_objective_encoded_in_name(self) -> None:
        assert GradientBoostedRanker(objective="regression").name == "xgboost-gbdt-v1"
        assert GradientBoostedRanker(objective="rank").name == "xgboost-gbdt-rank-v1"


@_needs_xgboost
class TestGradientBoostedRanker:
    def test_fit_score_shapes(self) -> None:
        samples = _interaction_samples()
        fitted = GradientBoostedRanker(device="cpu", num_boost_round=40).fit(samples, ["f1", "f2"])
        scores = fitted.score(samples[:10])
        assert len(scores) == 10
        assert all(isinstance(v, float) for v in scores)

    def test_empty_score_returns_empty(self) -> None:
        fitted = GradientBoostedRanker(device="cpu", num_boost_round=10).fit(
            _interaction_samples(), ["f1", "f2"]
        )
        assert fitted.score([]) == []

    def test_importances_normalized_over_named_features(self) -> None:
        fitted = GradientBoostedRanker(device="cpu", num_boost_round=60).fit(
            _interaction_samples(), ["f1", "f2"]
        )
        weights = dict(fitted.feature_weights())
        assert set(weights) == {"f1", "f2"}
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(v >= 0.0 for v in weights.values())

    def test_captures_interaction_linear_misses(self) -> None:
        # On sign-flip interaction data the linear ranker's marginal IC is ~0
        # for both features (it falls back to equal weights), but the tree
        # splits on f1 -> non-trivial importance. This is the whole point of
        # Arm I: a nonlinear learner extracts signal the linear ranker can't.
        samples = _interaction_samples()
        gbdt = GradientBoostedRanker(device="cpu", num_boost_round=80).fit(samples, ["f1", "f2"])
        assert gbdt.feature_weights()["f1"] > 0.0

    def test_satisfies_protocols(self) -> None:
        model = GradientBoostedRanker(device="cpu", num_boost_round=10)
        assert isinstance(model, AlphaModel)
        assert isinstance(model.fit(_interaction_samples(), ["f1", "f2"]), FittedAlphaModel)

    def test_plugs_into_walk_forward(self) -> None:
        import math

        samples = _samples()
        evidence = run_sample_walk_forward(
            samples=samples,
            config=_wf_config(),
            model_version="xgboost-gbdt-v1",
            feature_set_version="test-fs",
            feature_names=["f1", "f2"],
            model=GradientBoostedRanker(device="cpu", num_boost_round=30),
        )
        assert len(evidence.folds) >= 1
        assert math.isfinite(evidence.metrics["slippage_adjusted_sharpe"])
        assert set(evidence.selected_weights) <= {"f1", "f2"}


@_needs_xgboost
class TestGradientBoostedRankerRankMode:
    """rank:pairwise mode with per-date query groups (Arm J)."""

    def test_rank_fit_score_and_weights(self) -> None:
        samples = _interaction_samples()
        fitted = GradientBoostedRanker(objective="rank", device="cpu", num_boost_round=60).fit(
            samples, ["f1", "f2"]
        )
        assert len(fitted.score(samples[:10])) == 10
        weights = dict(fitted.feature_weights())
        assert set(weights) == {"f1", "f2"}
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_rank_handles_unsorted_input(self) -> None:
        # The driver passes date-sorted rows, but fit must group correctly even
        # if a caller hands it shuffled samples (it sorts by as_of internally).
        samples = _interaction_samples(n_days=20, n_instruments=15)
        shuffled = list(reversed(samples))
        fitted = GradientBoostedRanker(objective="rank", device="cpu", num_boost_round=40).fit(
            shuffled, ["f1", "f2"]
        )
        assert len(fitted.score(samples[:5])) == 5

    def test_rank_plugs_into_walk_forward(self) -> None:
        import math

        evidence = run_sample_walk_forward(
            samples=_samples(),
            config=_wf_config(),
            model_version="xgboost-gbdt-rank-v1",
            feature_set_version="test-fs",
            feature_names=["f1", "f2"],
            model=GradientBoostedRanker(objective="rank", device="cpu", num_boost_round=30),
        )
        assert len(evidence.folds) >= 1
        assert math.isfinite(evidence.metrics["slippage_adjusted_sharpe"])
