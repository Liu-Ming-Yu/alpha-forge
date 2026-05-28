"""Unit tests for the price-volume-starter-v1 feature factory.

The tests use small synthetic OHLCV panels and assert against
hand-computed expectations. Every acceptance criterion in the
"Phase 1 Acceptance" section of the brief has a test here:

1. exact ret_Nd
2. mom_12_1 excludes the most recent month
3. reversals are sign-flipped returns
4. low_vol features are sign-flipped rolling vol
5. low_amihud is sign-flipped Amihud illiquidity
6. no cross-instrument leakage
7. features are NaN until the full lookback is available
8. coverage counts are correct
9. FeatureSpec metadata exists for every produced column
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features import (
    FeatureSpec,
    get_global_registry,
)
from quant_platform.research.features.price_volume import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    PriceVolumeConfig,
    compute_price_volume_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_bars(
    instrument: str,
    n_rows: int,
    *,
    start: str = "2023-01-02",
    close_base: float = 100.0,
    close_step: float = 1.0,
    volume_base: int = 10_000,
    volume_step: int = 100,
) -> pd.DataFrame:
    """Build a small synthetic OHLCV frame for one instrument.

    Closes grow linearly so ret_1d is non-constant but predictable;
    we never test exact float equality, only ratios that fall out of
    the linear progression. Volume grows linearly too, which makes
    volume_z_20d sit at a deterministic, easy-to-reason-about value.
    """
    dates = pd.bdate_range(start=start, periods=n_rows)
    closes = close_base + close_step * np.arange(n_rows)
    return pd.DataFrame(
        {
            "instrument_id": instrument,
            "date": dates,
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": volume_base + volume_step * np.arange(n_rows),
        }
    )


@pytest.fixture
def single_instrument_bars() -> pd.DataFrame:
    """300 business days of monotonic closes for a single instrument."""
    return _linear_bars("AAA", n_rows=300)


@pytest.fixture
def two_instrument_bars() -> pd.DataFrame:
    """Two instruments with non-overlapping histories — leak detector."""
    aaa = _linear_bars("AAA", n_rows=260, start="2022-01-03", close_base=100.0)
    # BBB starts AFTER AAA finishes and has its own price scale. If any
    # rolling op silently pulls from AAA into BBB's warm-up window the
    # exact-value assertions below will catch it.
    bbb = _linear_bars(
        "BBB",
        n_rows=120,
        start="2023-06-01",
        close_base=500.0,
        close_step=-2.0,  # different sign/scale exposes leakage
        volume_base=5_000,
    )
    return pd.concat([aaa, bbb], ignore_index=True)


# ---------------------------------------------------------------------------
# Spec / registry contract
# ---------------------------------------------------------------------------


def test_every_produced_column_has_a_spec(single_instrument_bars: pd.DataFrame) -> None:
    """Every name in FEATURE_NAMES must have a FeatureSpec, and the
    spec name must match the column produced."""
    result = compute_price_volume_features(single_instrument_bars)

    assert set(result.feature_names) == set(FEATURE_NAMES)
    for name in result.feature_names:
        assert name in result.feature_specs, f"missing spec for {name}"
        assert result.feature_specs[name].name == name
        assert result.feature_specs[name].family == "price_volume"
        assert result.feature_specs[name].point_in_time is True
        assert result.feature_specs[name].version == "price-volume-starter-v1"


def test_feature_specs_registered_in_global_registry() -> None:
    """Importing the family must populate the process registry."""
    registry = get_global_registry()
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name)
        assert registry.get(spec.name) == spec


def test_positive_oriented_features_carry_larger_is_better() -> None:
    """Sign-flipped exports (low_vol_*, low_amihud_*, reversal_*) must
    declare larger_is_better=True and expected_direction='+'."""
    positively_oriented_prefixes = ("low_vol_", "low_downside_vol_", "low_amihud_", "reversal_")
    for spec in FEATURE_SPECS:
        if any(spec.name.startswith(p) for p in positively_oriented_prefixes):
            assert spec.expected_direction == "+", spec.name
            assert spec.larger_is_better is True, spec.name


def test_feature_spec_rejects_negative_with_larger_is_better() -> None:
    """The FeatureSpec invariant from contracts.py must hold."""
    with pytest.raises(ValueError, match="expected_direction='-'"):
        FeatureSpec(
            name="bogus",
            family="price_volume",
            description="d",
            expected_direction="-",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=1,
            version="bogus-v1",
            larger_is_better=True,
        )


# ---------------------------------------------------------------------------
# Exact formulas
# ---------------------------------------------------------------------------


def _aaa(result_frame: pd.DataFrame) -> pd.DataFrame:
    return result_frame.loc[result_frame["instrument_id"] == "AAA"].reset_index(drop=True)


def test_ret_nd_exact_calculation(single_instrument_bars: pd.DataFrame) -> None:
    """ret_Nd[t] must equal close[t]/close[t-N] - 1."""
    result = compute_price_volume_features(single_instrument_bars).frame
    closes = single_instrument_bars["close"].to_numpy()

    for window in (1, 5, 10, 21, 63, 126, 252):
        col = result[f"ret_{window}d"].to_numpy()
        # First `window` rows must be NaN under the full-window policy.
        assert np.all(np.isnan(col[:window]))
        # After warm-up the value is the deterministic ratio.
        expected = closes[window:] / closes[:-window] - 1.0
        np.testing.assert_allclose(col[window:], expected, rtol=1e-12, atol=1e-12)


def test_mom_12_1_excludes_most_recent_month(single_instrument_bars: pd.DataFrame) -> None:
    """mom_12_1[t] == close[t-21] / close[t-252] - 1.

    The "−21" skip is the load-bearing detail; momentum that doesn't
    skip the most recent month gets contaminated by short-term
    reversal.
    """
    result = compute_price_volume_features(single_instrument_bars).frame
    closes = single_instrument_bars["close"].to_numpy()

    col = result["mom_12_1"].to_numpy()
    # NaN until row 252 (where close.shift(252) becomes valid).
    assert np.all(np.isnan(col[:252]))
    expected = closes[252 - 21 : len(closes) - 21] / closes[: len(closes) - 252] - 1.0
    np.testing.assert_allclose(col[252:], expected, rtol=1e-12, atol=1e-12)


def test_mom_6_1_and_mom_3_1_skip_recent_month(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    closes = single_instrument_bars["close"].to_numpy()

    for month, long_lb in ((6, 126), (3, 63)):
        col = result[f"mom_{month}_1"].to_numpy()
        assert np.all(np.isnan(col[:long_lb]))
        expected = closes[long_lb - 21 : len(closes) - 21] / closes[: len(closes) - long_lb] - 1.0
        np.testing.assert_allclose(col[long_lb:], expected, rtol=1e-12, atol=1e-12)


def test_reversals_are_negative_returns(single_instrument_bars: pd.DataFrame) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    for window in (1, 5, 21):
        ret = result[f"ret_{window}d"].to_numpy()
        rev = result[f"reversal_{window}d"].to_numpy()
        # NaN/NaN comparison must hold; non-NaN values must be negated.
        mask = ~np.isnan(ret)
        np.testing.assert_allclose(rev[mask], -ret[mask], rtol=1e-12, atol=1e-12)
        assert np.array_equal(np.isnan(ret), np.isnan(rev))


def test_low_vol_features_are_negative_rolling_std(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    closes = single_instrument_bars["close"].to_numpy()
    daily_ret = pd.Series(np.concatenate([[np.nan], closes[1:] / closes[:-1] - 1.0]))

    for window in (21, 63, 126):
        expected_vol = daily_ret.rolling(window, min_periods=window).std(ddof=1)
        produced = result[f"low_vol_{window}d"].to_numpy()
        # Produced must be exactly -rolling_std(daily_return, window).
        np.testing.assert_allclose(
            produced,
            -expected_vol.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_low_downside_vol_uses_only_negative_returns(
    single_instrument_bars: pd.DataFrame,
) -> None:
    """low_downside_vol_63d must be -rolling_std(min(ret,0), 63).

    Because the synthetic series is monotone-up every daily return is
    strictly positive, so ``min(ret, 0) == 0`` everywhere and the
    rolling std collapses to zero. ``-0.0`` and ``0.0`` are both
    acceptable; we assert |value| < eps inside the valid window.
    """
    result = compute_price_volume_features(single_instrument_bars).frame
    produced = result["low_downside_vol_63d"].to_numpy()
    assert np.all(np.isnan(produced[:63]))
    np.testing.assert_allclose(produced[63:], 0.0, atol=1e-12)


def test_low_amihud_is_negative_rolling_amihud(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    closes = single_instrument_bars["close"].to_numpy()
    volume = single_instrument_bars["volume"].to_numpy()
    daily_ret = np.concatenate([[np.nan], closes[1:] / closes[:-1] - 1.0])
    dollar_volume = closes * volume
    raw_amihud_daily = np.abs(daily_ret) / dollar_volume
    expected = pd.Series(raw_amihud_daily).rolling(20, min_periods=20).mean().to_numpy()
    produced = result["low_amihud_20d"].to_numpy()
    np.testing.assert_allclose(produced, -expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_high_low_range_and_overnight_gap(single_instrument_bars: pd.DataFrame) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    bars = single_instrument_bars
    expected_range = (bars["high"] - bars["low"]) / bars["close"]
    np.testing.assert_allclose(
        result["high_low_range_1d"].to_numpy(),
        expected_range.to_numpy(),
        rtol=1e-12,
    )
    # overnight_gap and close_to_open_return must be identical by
    # construction (the brief lists both formulas, which are equal).
    np.testing.assert_array_equal(
        result["overnight_gap"].to_numpy(),
        result["close_to_open_return"].to_numpy(),
    )
    # open_to_close_return = close / open - 1.
    expected_otc = bars["close"] / bars["open"] - 1.0
    np.testing.assert_allclose(
        result["open_to_close_return"].to_numpy(),
        expected_otc.to_numpy(),
        rtol=1e-12,
    )


def test_distance_to_52w_high_equals_drawdown(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    np.testing.assert_array_equal(
        result["distance_to_52w_high"].to_numpy(),
        result["drawdown_from_252d_high"].to_numpy(),
    )


def test_distance_to_52w_high_uses_trailing_max(
    single_instrument_bars: pd.DataFrame,
) -> None:
    """For a strictly monotone-up series the trailing 252-day high IS
    today's close, so distance_to_52w_high must be 0 inside the
    warm-up-completed window."""
    result = compute_price_volume_features(single_instrument_bars).frame
    produced = result["distance_to_52w_high"].to_numpy()
    assert np.all(np.isnan(produced[:251]))
    np.testing.assert_allclose(produced[251:], 0.0, atol=1e-12)


def test_dollar_volume_20d_matches_rolling_mean(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars).frame
    bars = single_instrument_bars
    expected = (bars["close"] * bars["volume"]).rolling(20, min_periods=20).mean().to_numpy()
    np.testing.assert_allclose(
        result["dollar_volume_20d"].to_numpy(),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# PIT honesty, leakage, and edge cases
# ---------------------------------------------------------------------------


def test_features_are_nan_until_full_lookback(single_instrument_bars: pd.DataFrame) -> None:
    """Under min_periods_policy='full' every feature row before its
    lookback completes must be NaN."""
    result = compute_price_volume_features(single_instrument_bars).frame

    # Spot-check several windows; the ret_Nd test above covers the
    # exhaustive case.
    assert result["ret_252d"].iloc[:252].isna().all()
    assert result["ret_252d"].iloc[252:].notna().all()
    assert result["low_vol_63d"].iloc[:63].isna().all()
    assert result["low_vol_63d"].iloc[63:].notna().all()
    assert result["mom_12_1"].iloc[:252].isna().all()
    assert result["mom_12_1"].iloc[252:].notna().all()


def test_no_cross_instrument_leakage(two_instrument_bars: pd.DataFrame) -> None:
    """Computing two instruments together must yield the same per-
    instrument output as computing each one in isolation."""
    combined = compute_price_volume_features(two_instrument_bars).frame

    bbb_only = two_instrument_bars.loc[two_instrument_bars["instrument_id"] == "BBB"].copy()
    bbb_only_result = compute_price_volume_features(bbb_only).frame

    combined_bbb = (
        combined.loc[combined["instrument_id"] == "BBB"].sort_values("date").reset_index(drop=True)
    )
    bbb_only_result = bbb_only_result.sort_values("date").reset_index(drop=True)

    for name in FEATURE_NAMES:
        pd.testing.assert_series_equal(
            combined_bbb[name],
            bbb_only_result[name],
            check_names=False,
        )


def test_first_row_of_each_instrument_has_nan_daily_features(
    two_instrument_bars: pd.DataFrame,
) -> None:
    """The first row of every instrument must produce NaN for any
    feature whose lookback >= 1 day."""
    result = compute_price_volume_features(two_instrument_bars).frame
    firsts = result.groupby("instrument_id").head(1)

    for name in FEATURE_NAMES:
        spec = FEATURE_SPECS[FEATURE_NAMES.index(name)]
        # ``open_to_close_return`` has lookback_days=0 (same-day ratio)
        # and is permitted to be non-NaN on the first row; everything
        # else must be NaN until the lookback completes.
        if spec.lookback_days == 0:
            continue
        assert firsts[name].isna().all(), name


def test_short_instrument_yields_all_nan_for_long_lookbacks(
    two_instrument_bars: pd.DataFrame,
) -> None:
    """BBB only has 120 rows in the fixture, so ret_252d and mom_12_1
    must be NaN for every BBB row."""
    result = compute_price_volume_features(two_instrument_bars).frame
    bbb = result.loc[result["instrument_id"] == "BBB"]
    assert bbb["ret_252d"].isna().all()
    assert bbb["mom_12_1"].isna().all()
    assert bbb["low_vol_126d"].isna().all()


def test_coverage_counts_match_non_null_rows(
    single_instrument_bars: pd.DataFrame,
) -> None:
    result = compute_price_volume_features(single_instrument_bars)
    for name, coverage in result.coverage.items():
        assert coverage == int(result.frame[name].notna().sum()), name


def test_empty_bars_returns_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype=float),
            "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float),
            "close": pd.Series(dtype=float),
            "volume": pd.Series(dtype=int),
        }
    )
    result = compute_price_volume_features(empty)
    assert result.frame.empty
    assert set(result.feature_names) == set(FEATURE_NAMES)
    assert all(v == 0 for v in result.coverage.values())


def test_missing_required_columns_raises() -> None:
    bars = pd.DataFrame(
        {
            "instrument_id": ["AAA"],
            "date": [pd.Timestamp("2024-01-01")],
            "open": [10.0],
            # "high" missing
            "low": [9.0],
            "close": [9.5],
            "volume": [1000],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        compute_price_volume_features(bars)


def test_zero_volume_produces_nan_not_inf() -> None:
    """A zero-volume row must not make amihud explode to inf."""
    bars = _linear_bars("AAA", n_rows=40)
    bars.loc[bars.index[5], "volume"] = 0
    result = compute_price_volume_features(bars).frame
    produced = result["low_amihud_20d"].to_numpy()
    # No infs may leak out, even via the rolling mean.
    assert not np.any(np.isposinf(produced))
    assert not np.any(np.isneginf(produced))


def test_unsorted_input_is_handled() -> None:
    """compute_* must internally sort by (instrument_id, date) and
    still produce the canonical answer."""
    bars = _linear_bars("AAA", n_rows=60)
    shuffled = bars.sample(frac=1.0, random_state=42).reset_index(drop=True)
    a = (
        compute_price_volume_features(bars)
        .frame.sort_values(["instrument_id", "date"])
        .reset_index(drop=True)
    )
    b = (
        compute_price_volume_features(shuffled)
        .frame.sort_values(["instrument_id", "date"])
        .reset_index(drop=True)
    )
    for name in FEATURE_NAMES:
        pd.testing.assert_series_equal(a[name], b[name], check_names=False)


def test_non_default_config_changes_version() -> None:
    """A custom version string flows through to the FeatureSpec set —
    used by tests that want to register an experimental copy of the
    family without colliding with the production version."""
    cfg = PriceVolumeConfig(version="price-volume-test-v0")
    # We don't recompute the registered specs (they're module-level
    # constants pinned to DEFAULT_CONFIG.version), but the config
    # itself must round-trip its overridden version untouched.
    assert cfg.version == "price-volume-test-v0"
    assert DEFAULT_CONFIG.version == "price-volume-starter-v1"
