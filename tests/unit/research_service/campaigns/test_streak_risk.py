"""Tests for the fold-streak exposure throttle.

The contract pinned here:

1. PIT safety — the scale at fold N is a pure function of folds 0..N-1.
   The dial must never see the current fold's OOS IC; the caller appends
   that IC to history only after evaluation completes. Failing this
   silently makes the dial a look-ahead bias.
2. Warmup — during the first ``min_folds_before_active`` folds the
   scale is ``1.0`` and the reason is ``"warmup"``; no throttle.
3. Hard circuit breaker — when the trailing negative-IC streak reaches
   ``kill_streak`` the scale is ``0.0`` regardless of EWMA.
4. Soft EWMA throttle — linear interpolation from ``floor_ic`` (0) to
   ``ceiling_ic`` (1); above-ceiling caps at 1.0; below-floor caps at 0.0.
5. Config validation — out-of-range thresholds raise loudly at
   construction time so a misconfigured dial can't run silently.
6. Diagnostics payload — captures scale + reason + EWMA + streak so a
   reviewer can audit each fold's decision.
"""

from __future__ import annotations

import math

import pytest

from quant_platform.services.research_service.campaigns.portfolio.streak_risk import (
    FoldStreakRiskConfig,
    FoldStreakRiskScale,
    _ewma,
    _trailing_negative_streak,
    compute_fold_streak_exposure_scale,
    fold_streak_diagnostics_payload,
)


def _config(**overrides: object) -> FoldStreakRiskConfig:
    """Build a config with the latest-stack defaults plus per-test overrides."""
    base = {
        "min_folds_before_active": 4,
        "kill_streak": 3,
        "ewma_halflife": 4,
        "floor_ic": -0.02,
        "ceiling_ic": 0.0,
    }
    base.update(overrides)
    return FoldStreakRiskConfig(**base)  # type: ignore[arg-type]


# -- 1. PIT safety ---------------------------------------------------------


class TestPITSafety:
    """The scale at fold N must depend only on completed prior folds."""

    def test_scale_is_deterministic_in_prior_ics_only(self) -> None:
        # Same prior history → same scale, regardless of what comes
        # next. The function signature already enforces this (only
        # prior_fold_ics in, scale out), but pin it as behavior.
        cfg = _config()
        prior = (0.01, 0.005, -0.001, 0.002, 0.003)

        scale_a = compute_fold_streak_exposure_scale(prior, config=cfg)
        scale_b = compute_fold_streak_exposure_scale(prior, config=cfg)
        # Bit-for-bit equality, not just approx — pure function.
        assert scale_a.scale == scale_b.scale
        assert scale_a.reason == scale_b.reason
        assert scale_a.ewma_ic == scale_b.ewma_ic
        assert scale_a.neg_streak == scale_b.neg_streak

    def test_appending_a_future_ic_changes_the_next_scale_not_this_one(self) -> None:
        # Capture the scale at "fold 5" from prior 0..4. Then simulate
        # appending fold 5's IC and computing fold 6's scale. The
        # fold-5 result must be untouched.
        cfg = _config()
        prior_at_fold_5 = (0.01, 0.005, -0.001, 0.002, 0.003)
        fold_5_scale = compute_fold_streak_exposure_scale(prior_at_fold_5, config=cfg)

        # Simulate the caller appending the fold-5 IC and recomputing.
        prior_at_fold_6 = prior_at_fold_5 + (-0.005,)
        fold_6_scale = compute_fold_streak_exposure_scale(prior_at_fold_6, config=cfg)

        # Fold-5's scale must NOT have changed retroactively.
        fold_5_scale_recomputed = compute_fold_streak_exposure_scale(prior_at_fold_5, config=cfg)
        assert fold_5_scale.scale == fold_5_scale_recomputed.scale
        # And fold 6 sees one extra entry.
        assert fold_6_scale.prior_fold_count == fold_5_scale.prior_fold_count + 1


# -- 2. Warmup -------------------------------------------------------------


class TestWarmup:
    def test_zero_prior_folds_returns_warmup_scale_1(self) -> None:
        cfg = _config()
        scale = compute_fold_streak_exposure_scale((), config=cfg)
        assert scale.scale == 1.0
        assert scale.reason == "warmup"
        assert scale.prior_fold_count == 0

    def test_below_min_folds_returns_warmup_even_with_terrible_ic(self) -> None:
        cfg = _config(min_folds_before_active=4)
        # Three folds, all dramatically negative. Without warmup the
        # circuit breaker would fire (streak=3 ≥ kill_streak=3); the
        # warmup gate must override.
        scale = compute_fold_streak_exposure_scale((-0.10, -0.10, -0.10), config=cfg)
        assert scale.scale == 1.0
        assert scale.reason == "warmup"

    def test_exactly_min_folds_activates_the_dial(self) -> None:
        cfg = _config(min_folds_before_active=4, kill_streak=3)
        # Four folds, four negatives — past warmup, circuit breaker fires.
        scale = compute_fold_streak_exposure_scale((-0.10, -0.10, -0.10, -0.10), config=cfg)
        assert scale.scale == 0.0
        assert scale.reason == "circuit_breaker_kill"


# -- 3. Hard circuit breaker -----------------------------------------------


class TestCircuitBreaker:
    def test_kill_streak_three_fires_on_third_consecutive_negative(self) -> None:
        cfg = _config(min_folds_before_active=1, kill_streak=3)
        # Four positives then three negatives — streak = 3 → kill.
        scale = compute_fold_streak_exposure_scale(
            (0.01, 0.01, 0.01, 0.01, -0.001, -0.001, -0.001),
            config=cfg,
        )
        assert scale.scale == 0.0
        assert scale.reason == "circuit_breaker_kill"
        assert scale.neg_streak == 3

    def test_kill_streak_resets_on_positive_fold(self) -> None:
        cfg = _config(min_folds_before_active=1, kill_streak=3, ceiling_ic=0.0)
        # Two negatives then three positives — the trailing-negative
        # streak count resets to 0 even though we'd otherwise have
        # been at 2.
        scale = compute_fold_streak_exposure_scale(
            (-0.005, -0.005, 0.002, 0.003, 0.001),
            config=cfg,
        )
        assert scale.neg_streak == 0
        # Circuit breaker did NOT fire (the explicit point of this test).
        assert scale.reason != "circuit_breaker_kill"
        # EWMA is *near* zero — the early negatives still carry a small
        # decayed weight, so the throttle may sit just below 1.0 rather
        # than at the ceiling. The important contract here is "not killed",
        # not "exactly 1.0".
        assert scale.scale > 0.95

    def test_zero_ic_breaks_the_streak(self) -> None:
        # Convention: 0.0 IC is "no signal", not "negative signal".
        # It breaks the trailing-negative streak.
        cfg = _config(min_folds_before_active=1, kill_streak=2)
        scale = compute_fold_streak_exposure_scale((-0.01, 0.0, -0.01), config=cfg)
        assert scale.neg_streak == 1  # only the trailing entry counts


# -- 4. EWMA soft throttle -------------------------------------------------


class TestEWMAThrottle:
    def test_above_ceiling_caps_at_1(self) -> None:
        cfg = _config(min_folds_before_active=1)
        scale = compute_fold_streak_exposure_scale((0.05, 0.05, 0.05, 0.05, 0.05), config=cfg)
        assert scale.scale == 1.0
        assert scale.reason == "ewma_ceiling"

    def test_below_floor_caps_at_0(self) -> None:
        cfg = _config(min_folds_before_active=1, kill_streak=10)
        # All small negatives — EWMA roughly equals the mean of -0.05
        # which is way below the floor of -0.02. Scale = 0.
        scale = compute_fold_streak_exposure_scale((-0.05,), config=cfg)
        assert scale.scale == 0.0
        assert scale.reason == "ewma_floor"

    def test_between_floor_and_ceiling_linear_interpolates(self) -> None:
        # Halflife = 1 so the EWMA is dominated by the most-recent IC.
        # A single IC at -0.01 (halfway between floor -0.02 and ceiling 0.0)
        # should give scale = 0.5.
        cfg = _config(
            min_folds_before_active=1,
            kill_streak=10,
            ewma_halflife=1,
            floor_ic=-0.02,
            ceiling_ic=0.0,
        )
        scale = compute_fold_streak_exposure_scale((-0.01,), config=cfg)
        assert scale.reason == "ewma_throttled"
        assert scale.scale == pytest.approx(0.5, abs=1e-12)


# -- 5. Config validation --------------------------------------------------


class TestConfigValidation:
    def test_zero_min_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_folds_before_active"):
            FoldStreakRiskConfig(min_folds_before_active=0)

    def test_zero_kill_streak_rejected(self) -> None:
        with pytest.raises(ValueError, match="kill_streak"):
            FoldStreakRiskConfig(kill_streak=0)

    def test_zero_halflife_rejected(self) -> None:
        with pytest.raises(ValueError, match="ewma_halflife"):
            FoldStreakRiskConfig(ewma_halflife=0)

    def test_floor_ge_ceiling_rejected(self) -> None:
        with pytest.raises(ValueError, match="floor_ic must be < ceiling_ic"):
            FoldStreakRiskConfig(floor_ic=0.01, ceiling_ic=0.0)

    def test_nan_floor_rejected(self) -> None:
        with pytest.raises(ValueError, match="floor_ic and ceiling_ic must be finite"):
            FoldStreakRiskConfig(floor_ic=float("nan"))


# -- 6. FoldStreakRiskScale validation -------------------------------------


class TestScaleValidation:
    def test_scale_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale must be in"):
            FoldStreakRiskScale(
                scale=1.5,
                reason="ewma_ceiling",
                prior_fold_count=10,
                neg_streak=0,
                ewma_ic=0.01,
            )

    def test_nan_ewma_rejected(self) -> None:
        with pytest.raises(ValueError, match="ewma_ic must be finite"):
            FoldStreakRiskScale(
                scale=1.0,
                reason="warmup",
                prior_fold_count=0,
                neg_streak=0,
                ewma_ic=float("nan"),
            )


# -- 7. Internal helpers ---------------------------------------------------


class TestInternalHelpers:
    """Direct coverage of ``_trailing_negative_streak`` and ``_ewma``."""

    def test_streak_empty(self) -> None:
        assert _trailing_negative_streak(()) == 0

    def test_streak_all_negatives(self) -> None:
        assert _trailing_negative_streak((-0.01, -0.02, -0.03)) == 3

    def test_streak_breaks_at_positive(self) -> None:
        assert _trailing_negative_streak((-0.01, 0.01, -0.02, -0.03)) == 2

    def test_streak_breaks_at_zero(self) -> None:
        # 0.0 isn't < 0, so it breaks the streak.
        assert _trailing_negative_streak((-0.01, 0.0, -0.02)) == 1

    def test_streak_breaks_at_nan(self) -> None:
        assert _trailing_negative_streak((-0.01, float("nan"), -0.02)) == 1

    def test_ewma_empty(self) -> None:
        assert _ewma((), halflife=4) == 0.0

    def test_ewma_constant_input_returns_that_constant(self) -> None:
        assert _ewma((0.05,) * 10, halflife=4) == pytest.approx(0.05, abs=1e-12)

    def test_ewma_recent_dominates(self) -> None:
        # Single recent positive after many negatives → EWMA pulled toward
        # the recent value (but not all the way for halflife>1).
        result = _ewma((-0.01,) * 9 + (0.05,), halflife=2)
        # Most-recent value gets the largest weight (1.0); older entries
        # get exponentially smaller weight. So result > simple mean of
        # the whole series.
        simple_mean = (-0.01 * 9 + 0.05) / 10
        assert result > simple_mean

    def test_ewma_skips_nan_entries(self) -> None:
        # NaN contributes neither weight nor value.
        with_nan = _ewma((0.01, float("nan"), 0.01), halflife=4)
        without_nan = _ewma((0.01, 0.01), halflife=4)
        # The two should give the same result modulo the age shift
        # (NaN at the middle position skips a slot but the relative
        # ordering of the other entries is preserved).
        # Loose check: result should be near 0.01 in both cases.
        assert math.isfinite(with_nan)
        assert math.isfinite(without_nan)
        assert with_nan == pytest.approx(0.01, abs=1e-3)


# -- 8. Diagnostics payload ------------------------------------------------


class TestDiagnosticsPayload:
    def test_disabled_dial_returns_marker_payload(self) -> None:
        payload = fold_streak_diagnostics_payload(None, [])
        assert payload == {"config": None, "applied": False}

    def test_enabled_dial_returns_aggregate_plus_per_fold(self) -> None:
        cfg = _config()
        scales = [
            FoldStreakRiskScale(
                scale=1.0,
                reason="warmup",
                prior_fold_count=0,
                neg_streak=0,
                ewma_ic=0.0,
            ),
            FoldStreakRiskScale(
                scale=0.0,
                reason="circuit_breaker_kill",
                prior_fold_count=4,
                neg_streak=3,
                ewma_ic=-0.01,
            ),
            FoldStreakRiskScale(
                scale=0.5,
                reason="ewma_throttled",
                prior_fold_count=5,
                neg_streak=1,
                ewma_ic=-0.01,
            ),
        ]
        payload = fold_streak_diagnostics_payload(cfg, scales)
        assert payload["applied"] is True
        assert payload["n_folds"] == 3
        assert payload["scale_avg"] == pytest.approx(0.5, abs=1e-12)
        assert payload["scale_min"] == 0.0
        assert payload["zero_fold_count"] == 1
        assert len(payload["per_fold"]) == 3  # type: ignore[arg-type]
        # Config dump round-trips the fields.
        assert payload["config"]["kill_streak"] == cfg.kill_streak  # type: ignore[index]
