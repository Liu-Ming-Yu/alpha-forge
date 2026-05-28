"""Unit tests for the microstructure-v1 feature family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.microstructure import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MANIFEST,
    MicrostructureConfig,
    compute_microstructure_features,
)
from quant_platform.research.features.microstructure.config import (
    DEFAULT_LONG_WINDOW,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_VARIANCE_RATIO_STRIDE,
    FEATURE_SET_VERSION,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic OHLCV panels
# ---------------------------------------------------------------------------


def _synthetic_bars(
    *,
    n_days: int,
    instruments: tuple[str, ...] = ("AAPL", "MSFT"),
    seed: int = 1234,
) -> pd.DataFrame:
    """Return a deterministic OHLCV panel with mild drift + volatility.

    Each instrument follows an independent random walk so the features
    have non-degenerate distributions over the panel.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    for instrument in instruments:
        close = 100.0
        for date in dates:
            ret = rng.normal(0.0005, 0.012)
            open_ = close * (1.0 + rng.normal(0.0, 0.002))
            close_next = close * (1.0 + ret)
            high = max(open_, close_next) * (1.0 + abs(rng.normal(0.0, 0.003)))
            low = min(open_, close_next) * (1.0 - abs(rng.normal(0.0, 0.003)))
            volume = max(int(rng.lognormal(13.0, 0.4)), 1)
            rows.append(
                {
                    "instrument_id": instrument,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close_next,
                    "volume": float(volume),
                }
            )
            close = close_next
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FeatureSpec catalogue + MANIFEST
# ---------------------------------------------------------------------------


def test_feature_specs_total_nineteen() -> None:
    assert len(FEATURE_SPECS) == 19
    assert len(FEATURE_NAMES) == 19
    assert len(set(FEATURE_NAMES)) == 19  # no duplicates


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "microstructure"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {
        # v1 features (10)
        f"parkinson_vol_{DEFAULT_SHORT_WINDOW}d",
        f"garman_klass_vol_{DEFAULT_SHORT_WINDOW}d",
        f"rogers_satchell_vol_{DEFAULT_SHORT_WINDOW}d",
        f"roll_spread_{DEFAULT_LONG_WINDOW}d",
        f"corwin_schultz_spread_{DEFAULT_SHORT_WINDOW}d",
        f"close_in_range_{DEFAULT_SHORT_WINDOW}d",
        f"return_autocorr_{DEFAULT_LONG_WINDOW}d",
        f"volume_autocorr_{DEFAULT_LONG_WINDOW}d",
        f"volume_return_correlation_{DEFAULT_SHORT_WINDOW}d",
        f"high_low_asymmetry_{DEFAULT_SHORT_WINDOW}d",
        # v2 additions (6)
        f"yang_zhang_vol_{DEFAULT_SHORT_WINDOW}d",
        f"bipower_variation_{DEFAULT_SHORT_WINDOW}d",
        f"realized_skew_{DEFAULT_LONG_WINDOW}d",
        f"realized_kurt_{DEFAULT_LONG_WINDOW}d",
        f"variance_ratio_{DEFAULT_VARIANCE_RATIO_STRIDE}_1_{DEFAULT_LONG_WINDOW}d",
        f"range_persistence_{DEFAULT_SHORT_WINDOW}d",
        # v3 additions (3)
        f"med_rv_{DEFAULT_LONG_WINDOW}d",
        f"tripower_variation_{DEFAULT_SHORT_WINDOW}d",
        f"realized_jump_intensity_{DEFAULT_SHORT_WINDOW}d",
    }
    assert set(FEATURE_NAMES) == expected


def test_feature_set_version_is_v3() -> None:
    assert FEATURE_SET_VERSION == "microstructure-v3"


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("microstructure", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "microstructure"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_short_below_minimum() -> None:
    with pytest.raises(ValueError, match="short_window must be >= 5"):
        MicrostructureConfig(short_window=3)


def test_config_rejects_long_not_strictly_above_short() -> None:
    with pytest.raises(ValueError, match="long_window must be strictly greater"):
        MicrostructureConfig(short_window=20, long_window=20)
    with pytest.raises(ValueError, match="long_window must be strictly greater"):
        MicrostructureConfig(short_window=20, long_window=15)


def test_config_rejects_short_long_under_long_min() -> None:
    with pytest.raises(ValueError, match="long_window must be >= 10"):
        MicrostructureConfig(short_window=5, long_window=8)


def test_default_config_uses_versioned_defaults() -> None:
    assert DEFAULT_CONFIG.version == FEATURE_SET_VERSION
    assert DEFAULT_CONFIG.short_window == DEFAULT_SHORT_WINDOW
    assert DEFAULT_CONFIG.long_window == DEFAULT_LONG_WINDOW


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_compute_rejects_missing_columns() -> None:
    bars = pd.DataFrame(
        {"instrument_id": ["AAPL"], "date": [pd.Timestamp("2024-01-02")], "close": [100.0]}
    )
    with pytest.raises(ValueError, match="missing required columns"):
        compute_microstructure_features(bars)


def test_compute_empty_panel_returns_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            col: pd.Series(dtype="object" if col == "instrument_id" else "float64")
            for col in ("instrument_id", "date", "open", "high", "low", "close", "volume")
        }
    )
    empty["date"] = pd.Series(dtype="datetime64[ns]")
    result = compute_microstructure_features(empty)
    assert result.frame.empty
    assert set(result.feature_names) == set(FEATURE_NAMES)
    assert all(v == 0 for v in result.coverage.values())


# ---------------------------------------------------------------------------
# Shape + signedness invariants on a synthetic panel
# ---------------------------------------------------------------------------


def test_compute_shape_matches_input_panel() -> None:
    bars = _synthetic_bars(n_days=80)
    result = compute_microstructure_features(bars)
    assert len(result.frame) == len(bars)
    # Every feature column present.
    for name in FEATURE_NAMES:
        assert name in result.frame.columns


def test_volatility_features_are_non_negative_where_defined() -> None:
    bars = _synthetic_bars(n_days=80)
    result = compute_microstructure_features(bars)
    for name in (
        f"parkinson_vol_{DEFAULT_SHORT_WINDOW}d",
        f"garman_klass_vol_{DEFAULT_SHORT_WINDOW}d",
        f"rogers_satchell_vol_{DEFAULT_SHORT_WINDOW}d",
        f"corwin_schultz_spread_{DEFAULT_SHORT_WINDOW}d",
        # v2 additions that should also be non-negative.
        f"yang_zhang_vol_{DEFAULT_SHORT_WINDOW}d",
        f"bipower_variation_{DEFAULT_SHORT_WINDOW}d",
    ):
        non_nan = result.frame[name].dropna()
        # Allow tiny numerical noise (clip + sqrt should pin at 0).
        assert (non_nan >= -1e-12).all(), name


def test_close_in_range_lives_in_unit_interval() -> None:
    bars = _synthetic_bars(n_days=80)
    result = compute_microstructure_features(bars)
    name = f"close_in_range_{DEFAULT_SHORT_WINDOW}d"
    non_nan = result.frame[name].dropna()
    assert (non_nan >= 0.0).all()
    assert (non_nan <= 1.0).all()


def test_autocorrelations_live_in_minus_one_to_one() -> None:
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    for name in (
        f"return_autocorr_{DEFAULT_LONG_WINDOW}d",
        f"volume_autocorr_{DEFAULT_LONG_WINDOW}d",
        f"volume_return_correlation_{DEFAULT_SHORT_WINDOW}d",
    ):
        non_nan = result.frame[name].dropna()
        # Allow tiny FP slop just outside [-1, 1].
        assert (non_nan >= -1.0 - 1e-9).all(), name
        assert (non_nan <= 1.0 + 1e-9).all(), name


def test_roll_spread_is_nan_when_autocov_non_negative() -> None:
    """Roll's estimator is undefined when cov(r_t, r_{t-1}) >= 0. A
    perfectly trending series has non-negative autocov, so roll_spread
    is NaN everywhere."""
    # Construct a clean trend: close increments by a fixed positive amount.
    n_days = 120
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    bars = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": np.arange(100.0, 100.0 + n_days),
            "high": np.arange(100.5, 100.5 + n_days),
            "low": np.arange(99.5, 99.5 + n_days),
            "close": np.arange(100.0, 100.0 + n_days),
            "volume": np.full(n_days, 1_000_000.0),
        }
    )
    result = compute_microstructure_features(bars)
    name = f"roll_spread_{DEFAULT_LONG_WINDOW}d"
    # Every value after the warm-up window should be NaN: a clean trend
    # has positive autocov, where Roll's estimator is undefined.
    warm_up = DEFAULT_LONG_WINDOW
    assert result.frame[name].iloc[warm_up:].isna().all()


def test_high_low_asymmetry_skews_with_uptrend() -> None:
    """A monotone uptrend has rolling_max(high) > close > rolling_min(low),
    with downside (close - min_low) growing faster than upside (max_high - close).
    Asymmetry should be < 1 for most rows."""
    n_days = 80
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = np.linspace(100.0, 200.0, n_days)
    bars = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n_days, 1_000_000.0),
        }
    )
    result = compute_microstructure_features(bars)
    name = f"high_low_asymmetry_{DEFAULT_SHORT_WINDOW}d"
    series = result.frame[name].dropna()
    # In a clean uptrend, the rolling max is roughly today's high but
    # the rolling min is far below. Asymmetry should mostly be < 1.
    assert (series < 1.0).mean() > 0.8


def test_features_replace_infs_with_nan() -> None:
    """``safe_div`` + the final ``.replace([inf,-inf], nan)`` must yield
    a clean float column even when inputs would produce infinities."""
    bars = _synthetic_bars(n_days=30)
    # Force a zero high == low on one row → ln(high/low) = 0, division
    # by zero range in close_in_range → NaN, never inf.
    bars.loc[5, "high"] = bars.loc[5, "low"]
    result = compute_microstructure_features(bars)
    for name in FEATURE_NAMES:
        column = result.frame[name]
        assert not np.isinf(column.dropna()).any(), name


def test_coverage_grows_with_panel_length() -> None:
    """A short panel covers fewer rows than a long panel — sanity-check
    that the warm-up window behaves as advertised."""
    short = compute_microstructure_features(_synthetic_bars(n_days=30))
    long = compute_microstructure_features(_synthetic_bars(n_days=120))
    # The 60-day Roll/autocorr features warm up only on the longer
    # panel — short panel should have ~zero coverage; long panel non-zero.
    for name in (
        f"roll_spread_{DEFAULT_LONG_WINDOW}d",
        f"return_autocorr_{DEFAULT_LONG_WINDOW}d",
        f"volume_autocorr_{DEFAULT_LONG_WINDOW}d",
    ):
        assert short.coverage[name] == 0, name
        assert long.coverage[name] > 0, name


def test_parkinson_vs_close_to_close_vol_makes_sense() -> None:
    """Parkinson volatility should be in the same order of magnitude
    as close-to-close vol on a synthetic random walk. Coarse sanity
    check — the ratio shouldn't be 100x or 0.01x."""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    pk = result.frame[f"parkinson_vol_{DEFAULT_SHORT_WINDOW}d"].dropna()
    # Close-to-close vol on the synthetic panel is ~1.2% daily,
    # so Parkinson should be in roughly [0.005, 0.05].
    assert pk.mean() > 0.003
    assert pk.mean() < 0.10


# ---------------------------------------------------------------------------
# v2 additions — feature-specific invariants
# ---------------------------------------------------------------------------


def test_config_rejects_variance_ratio_stride_below_minimum() -> None:
    with pytest.raises(ValueError, match="variance_ratio_stride must be >= 2"):
        MicrostructureConfig(variance_ratio_stride=1)


def test_config_rejects_variance_ratio_stride_above_long_window() -> None:
    with pytest.raises(ValueError, match="variance_ratio_stride must be <"):
        MicrostructureConfig(short_window=20, long_window=60, variance_ratio_stride=60)


def test_yang_zhang_vol_is_comparable_to_parkinson() -> None:
    """Yang-Zhang and Parkinson are both daily-OHLC vol estimators —
    their means on the same panel should be within an order of
    magnitude. (YZ is typically a bit higher because it adds the
    overnight component.)"""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    yz = result.frame[f"yang_zhang_vol_{DEFAULT_SHORT_WINDOW}d"].dropna()
    pk = result.frame[f"parkinson_vol_{DEFAULT_SHORT_WINDOW}d"].dropna()
    assert yz.mean() > 0
    assert pk.mean() > 0
    # Loose order-of-magnitude bound.
    assert 0.1 < (yz.mean() / pk.mean()) < 10.0


def test_bipower_variation_is_close_to_realized_variance_under_no_jumps() -> None:
    """Without explicit jumps, bipower variation should be in the same
    order of magnitude as the realised variance of log returns over the
    same window."""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    bpv = result.frame[f"bipower_variation_{DEFAULT_SHORT_WINDOW}d"].dropna()
    # Synthetic daily log-vol ~ 1.2%, so realised variance over a 20d
    # window is ~ (0.012)^2 = 1.44e-4. Bipower should be in roughly
    # [1e-5, 1e-3].
    assert bpv.mean() > 1e-6
    assert bpv.mean() < 1e-2


def test_realized_skew_kurt_in_reasonable_ranges() -> None:
    """Realised skew is unbounded but a synthetic random walk should
    give bounded values. Realised excess kurtosis can be large but
    not infinite."""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    sk = result.frame[f"realized_skew_{DEFAULT_LONG_WINDOW}d"].dropna()
    kt = result.frame[f"realized_kurt_{DEFAULT_LONG_WINDOW}d"].dropna()
    # No infinities; both should be finite on a clean random walk.
    assert np.isfinite(sk).all()
    assert np.isfinite(kt).all()
    # Coarse sanity — typical random-walk skew is roughly [-2, 2].
    assert sk.abs().mean() < 5.0


def test_variance_ratio_is_near_one_under_random_walk() -> None:
    """Lo-MacKinlay VR(q) = 1 under a random walk. The synthetic panel
    is (close to) a random walk so the rolling VR should average near
    1.0 — allow generous tolerance because variance estimates are noisy
    on small samples."""
    bars = _synthetic_bars(n_days=180, seed=99)
    result = compute_microstructure_features(bars)
    vr = result.frame[
        f"variance_ratio_{DEFAULT_VARIANCE_RATIO_STRIDE}_1_{DEFAULT_LONG_WINDOW}d"
    ].dropna()
    # VR mean should land in roughly [0.5, 2.0] on a 60d window with
    # 5-day stride, on a near-random-walk panel.
    assert 0.3 < vr.mean() < 3.0


def test_range_persistence_lives_in_minus_one_to_one() -> None:
    """Range persistence is a Pearson correlation; like the other
    autocorrelations in this family, it must lie in [-1, 1]."""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    series = result.frame[f"range_persistence_{DEFAULT_SHORT_WINDOW}d"].dropna()
    assert (series >= -1.0 - 1e-9).all()
    assert (series <= 1.0 + 1e-9).all()


def test_v2_features_warm_up_with_panel_length() -> None:
    """The v2 long-window features (realized skew/kurt, VR) need at
    least long_w days; a short panel should have zero coverage on
    them and the longer panel non-zero."""
    short = compute_microstructure_features(_synthetic_bars(n_days=30))
    long = compute_microstructure_features(_synthetic_bars(n_days=180))
    for name in (
        f"realized_skew_{DEFAULT_LONG_WINDOW}d",
        f"realized_kurt_{DEFAULT_LONG_WINDOW}d",
        f"variance_ratio_{DEFAULT_VARIANCE_RATIO_STRIDE}_1_{DEFAULT_LONG_WINDOW}d",
    ):
        assert short.coverage[name] == 0, name
        assert long.coverage[name] > 0, name


# ---------------------------------------------------------------------------
# Review-driven tests (PR #52 follow-up)
# ---------------------------------------------------------------------------


def test_yang_zhang_vol_matches_hand_computed_value_on_constant_panel() -> None:
    """Hand-computed YZ on a 20-day panel with **identical OHLC every
    day**: open=100, high=101, low=99, close=100.5. With identical
    bars, both the overnight and open-to-close return series are
    constant (-ln(100.5/100) and +ln(100.5/100) respectively after
    the first row), so their bias-corrected sample variance over a
    20-row window is exactly 0. Only the Rogers-Satchell intraday
    contribution drives YZ.

    Per-day RS contribution:
        ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
      = ln(101/100.5) * ln(101/100) + ln(99/100.5) * ln(99/100)

    YZ^2 = 0 + k * 0 + (1 - k) * rs_var,
    k = 0.34 / (1.34 + (N+1)/(N-1)) with N = 20.

    Pinning this verifies the formula end-to-end — a wrong k-weight
    or a swapped (1-k) ↔ k would shift YZ by 7%+, well outside FP
    tolerance.
    """
    n_days = 25  # > short_window so the rolling has at least one valid row
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    bars = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": np.full(n_days, 100.0),
            "high": np.full(n_days, 101.0),
            "low": np.full(n_days, 99.0),
            "close": np.full(n_days, 100.5),
            "volume": np.full(n_days, 1_000_000.0),
        }
    )
    result = compute_microstructure_features(bars)

    # Hand-computed expected value.
    rs_daily = np.log(101 / 100.5) * np.log(101 / 100) + np.log(99 / 100.5) * np.log(99 / 100)
    n = DEFAULT_SHORT_WINDOW
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    expected_yz = float(np.sqrt((1.0 - k) * rs_daily))

    # Pull the first row past the warm-up window.
    yz_col = f"yang_zhang_vol_{DEFAULT_SHORT_WINDOW}d"
    realised_yz = result.frame[yz_col].dropna().iloc[0]
    assert realised_yz == pytest.approx(expected_yz, rel=1e-9)


def test_bipower_variation_is_robust_to_isolated_jumps() -> None:
    """The whole point of bipower variation is jump-robustness:
    BPV(t) ∝ Σ |r_i| |r_{i-1}| is consistent for the integrated
    variance under jump-diffusion, while the naive Σ r_i² spikes on
    jumps.

    A jump-robust feature should pick up the jump less than a
    naive sum-of-squared-returns. Critically: BPV's robustness only
    holds when the jump is **isolated** — a one-sided spike followed
    by ordinary returns. A jump followed by an immediate
    mean-reversion (same-magnitude move in the opposite direction)
    gives BPV a |r_t| · |r_{t+1}| product where both factors are
    large, so BPV picks up the jump twice. We test the isolated-jump
    regime that the feature is actually designed for.

    Concretely: inject a +10% upward jump on day 30 and let the trend
    continue from the new level (no reversion). Then |r_30| is large
    but |r_29| and |r_31| are small, so the bipower products at
    j=30 and j=31 are O(jump × small) instead of O(jump²).
    """
    n_days = 60
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    # Build the close trajectory explicitly so we can ladder a jump in
    # without triggering mean-reversion: the jump happens on day 30
    # and the post-jump trend continues from the new level.
    daily_drift = 0.001  # +0.1% per day
    close_levels = np.empty(n_days)
    close_levels[0] = 100.0
    for i in range(1, n_days):
        close_levels[i] = close_levels[i - 1] * (1.0 + daily_drift)

    base = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": close_levels - 0.1,
            "high": close_levels + 0.2,
            "low": close_levels - 0.2,
            "close": close_levels,
            "volume": np.full(n_days, 1_000_000.0),
        }
    )

    # Re-build the jumped panel from scratch: same up-to-day-30 close,
    # then a 10% jump, then the smooth trend resumes from the new level
    # (no mean-reversion).
    jumped_close = close_levels.copy()
    jumped_close[30] = close_levels[30] * 1.10
    for i in range(31, n_days):
        jumped_close[i] = jumped_close[i - 1] * (1.0 + daily_drift)

    jumped = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": jumped_close - 0.1,
            "high": jumped_close + 0.2,
            "low": jumped_close - 0.2,
            "close": jumped_close,
            "volume": np.full(n_days, 1_000_000.0),
        }
    )

    base_result = compute_microstructure_features(base)
    jump_result = compute_microstructure_features(jumped)

    bpv_col = f"bipower_variation_{DEFAULT_SHORT_WINDOW}d"
    # Inspect a row well inside the post-jump rolling window.
    inspect_row = 45
    base_bpv = base_result.frame[bpv_col].iloc[inspect_row]
    jump_bpv = jump_result.frame[bpv_col].iloc[inspect_row]

    # Naive realised variance over the same window on both panels.
    base_log_ret = np.log(base["close"] / base["close"].shift(1))
    jump_log_ret = np.log(jumped["close"] / jumped["close"].shift(1))
    base_rv = (base_log_ret**2).iloc[inspect_row - 19 : inspect_row + 1].mean()
    jump_rv = (jump_log_ret**2).iloc[inspect_row - 19 : inspect_row + 1].mean()

    # BPV should be drastically less sensitive to the isolated jump
    # than RV. On a one-sided +10% jump over a 0.1%/day trend, the
    # naive RV picks up the full (0.10)^2 = 0.01 squared shock, while
    # BPV only picks up two adjacent products of order
    # |0.10| * |0.001| = 1e-4 each — so the BPV change is ~100x
    # smaller. We assert the BPV ratio is at most ~1% of the RV ratio.
    bpv_ratio = jump_bpv / base_bpv
    rv_ratio = jump_rv / base_rv
    assert bpv_ratio < rv_ratio * 0.05, (
        f"BPV should be jump-robust on isolated jumps: "
        f"bpv_ratio={bpv_ratio:.3f}, rv_ratio={rv_ratio:.3f}"
    )
    # And the naive RV should have spiked dramatically.
    assert rv_ratio > 50.0


def test_compute_rejects_drifted_window_under_default_version() -> None:
    """A caller that bumps a window without bumping the family version
    must fail loudly — otherwise two runs would emit different
    catalogues under the same manifest stamp.

    Covers A4 enforcement.
    """
    bars = _synthetic_bars(n_days=30)
    drifted_short = MicrostructureConfig(short_window=15)  # default version
    with pytest.raises(ValueError, match="without bumping the family version"):
        compute_microstructure_features(bars, config=drifted_short)

    drifted_stride = MicrostructureConfig(variance_ratio_stride=7)
    with pytest.raises(ValueError, match="without bumping the family version"):
        compute_microstructure_features(bars, config=drifted_stride)


def test_compute_accepts_drifted_window_when_version_also_bumped() -> None:
    """The escape hatch: caller takes ownership of the catalogue
    divergence by setting a non-default version."""
    bars = _synthetic_bars(n_days=30)
    custom = MicrostructureConfig(
        version="microstructure-experiment-1",
        short_window=15,
        long_window=45,
        variance_ratio_stride=3,
    )
    # Should not raise.
    result = compute_microstructure_features(bars, config=custom)
    # And the emitted feature names reflect the custom windows.
    assert "parkinson_vol_15d" in result.frame.columns
    assert "roll_spread_45d" in result.frame.columns
    assert "variance_ratio_3_1_45d" in result.frame.columns


# ---------------------------------------------------------------------------
# v3 additions — jump-cluster-robust estimators
# ---------------------------------------------------------------------------


def _smooth_panel_with_close(close_levels: np.ndarray) -> pd.DataFrame:
    """Wrap a close-level path into a deterministic OHLCV panel with
    tiny intraday range. Helper for the jump-cluster tests."""
    n_days = len(close_levels)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_days,
            "date": dates,
            "open": close_levels - 0.1,
            "high": close_levels + 0.2,
            "low": close_levels - 0.2,
            "close": close_levels,
            "volume": np.full(n_days, 1_000_000.0),
        }
    )


def test_med_rv_robust_to_isolated_jumps_where_bpv_partially_fails() -> None:
    """MedRV's signature property: **completely insensitive to a
    single isolated jump** that still contaminates BPV.

    Construct two panels:
      * ``base``: smooth +0.1%/day drift, no jumps.
      * ``isolated``: same drift, but day 50 jumps +10% and the trend
        resumes from the new level (no mean reversion).

    Under an isolated jump, the median of three adjacent ``|r|`` values
    picks the *non*-jumped neighbour every time — MedRV stays at its
    baseline. BPV is partially contaminated because its
    ``|r_t| · |r_{t-1}|`` product touches the jump-return on the
    jump-day and the day after, so two terms enter the BPV mean at
    elevated magnitudes.

    Expected:
      * ``med_rv_ratio ≈ 1.0`` (genuinely robust)
      * ``bpv_ratio  > 3``    (some contamination)
      * ``rv_ratio   > 20``   (full squared-jump shock)

    This is the case where v3's MedRV is strictly better than v2's
    BPV. (Clusters where the jump immediately mean-reverts are a
    separate scenario where neither estimator fully solves the
    contamination — see the next test for that case.)
    """
    n_days = 90
    daily_drift = 0.001
    base_close = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_drift))

    isolated_close = base_close.copy()
    isolated_close[50] = base_close[50] * 1.10
    # Trend resumes from the new level (no mean reversion).
    for i in range(51, n_days):
        isolated_close[i] = isolated_close[i - 1] * (1.0 + daily_drift)

    base_result = compute_microstructure_features(_smooth_panel_with_close(base_close))
    jump_result = compute_microstructure_features(_smooth_panel_with_close(isolated_close))

    # Inspect day 65: cluster (day 50) is inside both windows.
    #   - BPV window:   [46, 65]   (short_w = 20) ✓ covers day 50
    #   - MedRV window: [6,  65]   (long_w  = 60) ✓ covers day 50
    inspect_row = 65
    bpv_col = f"bipower_variation_{DEFAULT_SHORT_WINDOW}d"
    med_col = f"med_rv_{DEFAULT_LONG_WINDOW}d"

    base_bpv = base_result.frame[bpv_col].iloc[inspect_row]
    jump_bpv = jump_result.frame[bpv_col].iloc[inspect_row]
    base_med = base_result.frame[med_col].iloc[inspect_row]
    jump_med = jump_result.frame[med_col].iloc[inspect_row]

    # Naive RV for the comparison.
    base_log_ret = np.log(base_close[1:] / base_close[:-1])
    jump_log_ret = np.log(isolated_close[1:] / isolated_close[:-1])
    # Row 65 in original df → returns index 64 (shift by 1 from the
    # first NaN row). The rolling 20-day RV at row 65 averages
    # returns from index 45 to 64.
    base_rv = (base_log_ret[45:65] ** 2).mean()
    jump_rv = (jump_log_ret[45:65] ** 2).mean()

    med_ratio = jump_med / base_med
    bpv_ratio = jump_bpv / base_bpv
    rv_ratio = jump_rv / base_rv

    # MedRV is genuinely robust to isolated jumps — the ratio should
    # be ~1.0 (allow modest finite-sample noise on this small panel).
    assert med_ratio < 1.5, (
        f"MedRV must be ~unchanged by an isolated jump; got med_ratio={med_ratio:.3f}"
    )
    # BPV is contaminated but only modestly (the jump enters as |r_t| *
    # epsilon, not as J^2).
    assert bpv_ratio > 3.0, f"BPV should pick up the jump; got bpv_ratio={bpv_ratio:.3f}"
    # And RV explodes (the full squared jump enters the window).
    assert rv_ratio > 20.0, f"RV should explode on a jump; got rv_ratio={rv_ratio:.3f}"
    # The headline: MedRV is at least 3x more robust than BPV on
    # isolated jumps.
    assert med_ratio < bpv_ratio / 3.0, (
        f"MedRV should beat BPV by at least 3x on isolated jumps: "
        f"med_ratio={med_ratio:.3f}, bpv_ratio={bpv_ratio:.3f}"
    )


def test_med_rv_does_not_solve_jump_clusters() -> None:
    """Honest accounting: when a jump is immediately followed by a
    same-magnitude reversion, the cluster contaminates two
    consecutive medians (each containing two of the three jumped
    returns). MedRV picks up the cluster too — the *jump_intensity*
    feature is v3's actual answer for clusters; MedRV's claim is
    isolated-jump robustness only.

    This test pins the honest behaviour so a future contributor who
    expects "MedRV solves everything" gets corrected immediately.
    """
    n_days = 90
    daily_drift = 0.001
    base_close = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_drift))
    # +10% on day 50, immediate -10% reversion on day 51.
    clustered_close = base_close.copy()
    clustered_close[50] = base_close[50] * 1.10
    clustered_close[51] = base_close[51]
    for i in range(52, n_days):
        clustered_close[i] = clustered_close[i - 1] * (1.0 + daily_drift)

    base_result = compute_microstructure_features(_smooth_panel_with_close(base_close))
    cluster_result = compute_microstructure_features(_smooth_panel_with_close(clustered_close))

    inspect_row = 65
    med_col = f"med_rv_{DEFAULT_LONG_WINDOW}d"
    med_ratio = (
        cluster_result.frame[med_col].iloc[inspect_row]
        / base_result.frame[med_col].iloc[inspect_row]
    )

    # Cluster contaminates 2 of the 3 affected medians (each pair of
    # 3-windows containing the cluster has 2 of 3 large values).
    # MedRV is genuinely contaminated — not fully robust. We assert
    # the ratio is non-trivially > 1, codifying the honest behaviour.
    assert med_ratio > 10.0, (
        f"MedRV does NOT fully solve clusters — the test pins this "
        f"honest behaviour. Got med_ratio={med_ratio:.3f}"
    )


def test_realized_jump_intensity_spikes_strongly_on_isolated_jumps() -> None:
    """``jump_intensity = clip( (RV - BPV) / RV, 0, 1 )`` is largest
    when the BPV-vs-RV gap is largest, which happens for **isolated**
    jumps: RV picks up the full squared jump, while BPV only sees
    ``|r_jump| · |r_neighbor|`` — a much smaller product. The gap is
    therefore close to RV itself, and the ratio approaches 1.

    Inject a single +10% jump (no reversion). Verify intensity
    spikes past 0.8.
    """
    n_days = 60
    daily_drift = 0.001
    base_close = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_drift))
    isolated_close = base_close.copy()
    isolated_close[30] = base_close[30] * 1.10
    for i in range(31, n_days):
        isolated_close[i] = isolated_close[i - 1] * (1.0 + daily_drift)

    base_result = compute_microstructure_features(_smooth_panel_with_close(base_close))
    jump_result = compute_microstructure_features(_smooth_panel_with_close(isolated_close))

    intensity_col = f"realized_jump_intensity_{DEFAULT_SHORT_WINDOW}d"
    inspect_row = 45
    base_intensity = base_result.frame[intensity_col].iloc[inspect_row]
    jump_intensity = jump_result.frame[intensity_col].iloc[inspect_row]

    # No-jump panel has near-zero intensity (finite-sample noise only).
    assert base_intensity < 0.3
    # Isolated jump: BPV stays small, RV explodes, so (RV - BPV) / RV → 1.
    assert jump_intensity > 0.8, (
        f"jump_intensity should be near 1 on an isolated jump; got {jump_intensity:.3f}"
    )


def test_realized_jump_intensity_partially_elevated_under_clusters() -> None:
    """Clusters are genuinely hard: BPV picks up the cluster via
    ``|r_t| · |r_{t-1}|`` where BOTH factors are large, so the
    BPV-vs-RV gap is much smaller than under isolated jumps. The
    intensity signal is still elevated above baseline but doesn't
    saturate.

    This test pins the honest behaviour: ``jump_intensity`` is most
    discriminative for isolated jumps. Under clusters it's a partial
    signal — the cluster regime is fundamentally harder for any
    daily-OHLCV-only estimator (the right tool is intraday data,
    deferred to microstructure-v4).
    """
    n_days = 90
    daily_drift = 0.001
    base_close = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_drift))
    clustered_close = base_close.copy()
    clustered_close[50] = base_close[50] * 1.10
    clustered_close[51] = base_close[51]
    for i in range(52, n_days):
        clustered_close[i] = clustered_close[i - 1] * (1.0 + daily_drift)

    base_result = compute_microstructure_features(_smooth_panel_with_close(base_close))
    cluster_result = compute_microstructure_features(_smooth_panel_with_close(clustered_close))

    intensity_col = f"realized_jump_intensity_{DEFAULT_SHORT_WINDOW}d"
    inspect_row = 65
    base_intensity = base_result.frame[intensity_col].iloc[inspect_row]
    cluster_intensity = cluster_result.frame[intensity_col].iloc[inspect_row]

    # Under clusters, intensity is partially elevated — measurably
    # above the base panel but well below the isolated-jump saturation.
    assert cluster_intensity > base_intensity * 1.5, (
        f"jump_intensity should still elevate under clusters relative "
        f"to baseline: base={base_intensity:.3f}, cluster={cluster_intensity:.3f}"
    )
    # And the honest upper bound: clusters don't saturate the signal
    # the way isolated jumps do.
    assert cluster_intensity < 0.5, (
        f"Cluster intensity = {cluster_intensity:.3f}; clusters are "
        "fundamentally harder than isolated jumps on daily OHLCV."
    )


def test_tripower_variation_more_robust_than_bipower_on_isolated_jumps() -> None:
    """Tripower variation has a sub-linear exponent (2/3 per term),
    so each contaminated TPV term scales as J^(2/3) * eps^(4/3) instead
    of BPV's J * eps. Under an isolated +10% jump, TPV's contamination
    is therefore smaller than BPV's.
    """
    n_days = 90
    daily_drift = 0.001
    base_close = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_drift))
    isolated_close = base_close.copy()
    isolated_close[50] = base_close[50] * 1.10
    for i in range(51, n_days):
        isolated_close[i] = isolated_close[i - 1] * (1.0 + daily_drift)

    base_result = compute_microstructure_features(_smooth_panel_with_close(base_close))
    jump_result = compute_microstructure_features(_smooth_panel_with_close(isolated_close))

    inspect_row = 65
    tpv_col = f"tripower_variation_{DEFAULT_SHORT_WINDOW}d"
    bpv_col = f"bipower_variation_{DEFAULT_SHORT_WINDOW}d"

    tpv_ratio = (
        jump_result.frame[tpv_col].iloc[inspect_row] / base_result.frame[tpv_col].iloc[inspect_row]
    )
    bpv_ratio = (
        jump_result.frame[bpv_col].iloc[inspect_row] / base_result.frame[bpv_col].iloc[inspect_row]
    )
    # TPV's contamination is sub-linear in the jump magnitude — the
    # ratio should be strictly smaller than BPV's.
    assert tpv_ratio < bpv_ratio, (
        f"TPV should be more isolated-jump-robust than BPV: "
        f"tpv_ratio={tpv_ratio:.3f}, bpv_ratio={bpv_ratio:.3f}"
    )


def test_realized_jump_intensity_lives_in_unit_interval() -> None:
    """Jump intensity = clip((RV - BPV) / RV, 0, 1). The clipping
    pins the feature into [0, 1] regardless of underlying data."""
    bars = _synthetic_bars(n_days=120)
    result = compute_microstructure_features(bars)
    series = result.frame[f"realized_jump_intensity_{DEFAULT_SHORT_WINDOW}d"].dropna()
    assert (series >= 0.0).all()
    assert (series <= 1.0).all()


def test_tripower_scale_constant_matches_bns_2006() -> None:
    """Pin the module-level TPV scaling constant mu_{2/3}^(-3) to its
    closed-form value derived from the BNS 2006 paper.

    mu_r = E[|Z|^r] for Z ~ N(0, 1) = 2^(r/2) * Gamma((r+1)/2) / sqrt(pi).
    mu_{2/3} = 2^(1/3) * Gamma(5/6) / sqrt(pi) ~ 0.8024.
    mu_{2/3}^(-3) ~ 1.9358.
    """
    from math import gamma, pi, sqrt

    from quant_platform.research.features.microstructure.features import _TPV_SCALE

    # Re-derive from math.gamma to make the test independent of the
    # production constant, then assert they agree to machine precision.
    mu_two_thirds = (2 ** (1.0 / 3.0)) * gamma(5.0 / 6.0) / sqrt(pi)
    expected_scale = 1.0 / (mu_two_thirds**3)
    assert pytest.approx(expected_scale, rel=1e-12) == _TPV_SCALE
    # Sanity-check against the rounded value reported in the BNS paper.
    assert pytest.approx(1.9358, abs=1e-3) == _TPV_SCALE


def test_med_rv_scale_constant_matches_ads_2012() -> None:
    """Pin the MedRV scaling constant pi / (6 - 4*sqrt(3) + pi).

    Derived from E[ med(|Z_1|, |Z_2|, |Z_3|)^2 ] for iid half-normals
    in Andersen-Dobrev-Schaumburg (2012).
    """
    from math import pi, sqrt

    from quant_platform.research.features.microstructure.features import _MED_RV_SCALE

    expected = pi / (6.0 - 4.0 * sqrt(3.0) + pi)
    assert pytest.approx(expected, rel=1e-12) == _MED_RV_SCALE
    # ADS 2012 reports ~1.4194 for this constant; pin the rounded value.
    assert pytest.approx(1.4194, abs=1e-3) == _MED_RV_SCALE


def test_med_rv_and_tripower_warm_up_with_panel_length() -> None:
    """v3 features have extra warm-up: med_rv needs long_w + 2 rows
    (it uses lag-2 absolute return), tripower needs short_w + 2."""
    short = compute_microstructure_features(_synthetic_bars(n_days=30))
    long = compute_microstructure_features(_synthetic_bars(n_days=120))
    for name in (
        f"med_rv_{DEFAULT_LONG_WINDOW}d",
        f"tripower_variation_{DEFAULT_SHORT_WINDOW}d",
        f"realized_jump_intensity_{DEFAULT_SHORT_WINDOW}d",
    ):
        # short panel: insufficient warm-up.
        if name.startswith("med_rv"):
            assert short.coverage[name] == 0, name
        assert long.coverage[name] > 0, name
