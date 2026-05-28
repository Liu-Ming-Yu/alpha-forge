"""Unit tests for the macro-v1 feature family."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.macro import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FRED_CORPORATE_AAA,
    FRED_CORPORATE_BAA,
    FRED_DOLLAR_INDEX,
    FRED_TIPS_10Y,
    FRED_TREASURY_2Y,
    FRED_TREASURY_3M,
    FRED_TREASURY_10Y,
    FRED_VIX,
    MANIFEST,
    REQUIRED_SERIES_IDS,
    MacroConfig,
    MacroSeriesValue,
    compute_macro_features,
)
from quant_platform.research.features.macro.aggregator import (
    AggregatedMacroPanel,
    build_macro_panel,
)
from quant_platform.research.features.macro.config import (
    DEFAULT_DOLLAR_INDEX_WINDOW_DAYS,
    FEATURE_SET_VERSION,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _trading_dates(start: datetime, n_days: int) -> pd.DatetimeIndex:
    naive_start = start.replace(tzinfo=None)
    return pd.date_range(naive_start, periods=n_days, freq="B")


def _series_value(series_id: str, date: datetime, value: float) -> MacroSeriesValue:
    return MacroSeriesValue(series_id=series_id, observation_date=date, value=value)


def _full_series_set(*, observation_date: datetime, **overrides: float) -> list[MacroSeriesValue]:
    """Build one snapshot of all 8 required FRED series with default
    values; overrides let individual tests customise one or two."""
    defaults = {
        FRED_TREASURY_10Y: 4.50,
        FRED_TREASURY_2Y: 4.80,  # inverted by default (10y < 2y)
        FRED_TREASURY_3M: 5.00,
        FRED_CORPORATE_BAA: 6.20,
        FRED_CORPORATE_AAA: 5.40,
        FRED_VIX: 18.0,
        FRED_DOLLAR_INDEX: 105.0,
        FRED_TIPS_10Y: 1.80,
    }
    defaults.update(overrides)
    return [
        _series_value(series_id=sid, date=observation_date, value=val)
        for sid, val in defaults.items()
    ]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_series_value_rejects_naive_date() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MacroSeriesValue(series_id="DGS10", observation_date=datetime(2024, 6, 1), value=4.5)


def test_series_value_rejects_nan() -> None:
    with pytest.raises(ValueError, match="must be a finite float"):
        MacroSeriesValue(series_id="DGS10", observation_date=_utc(2024, 6, 1), value=float("nan"))


def test_series_value_rejects_empty_series_id() -> None:
    with pytest.raises(ValueError, match="series_id must be non-empty"):
        MacroSeriesValue(series_id="   ", observation_date=_utc(2024, 6, 1), value=4.5)


# ---------------------------------------------------------------------------
# Catalogue + manifest
# ---------------------------------------------------------------------------


def test_feature_specs_total_six() -> None:
    assert len(FEATURE_SPECS) == 6
    assert len(FEATURE_NAMES) == 6
    assert len(set(FEATURE_NAMES)) == 6


def test_feature_set_version_is_v1() -> None:
    assert FEATURE_SET_VERSION == "macro-v1"


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "macro"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {
        "yield_curve_slope_10y_2y",
        "yield_curve_slope_10y_3m",
        "credit_spread_baa_aaa",
        "vix_level",
        f"dollar_index_change_{DEFAULT_DOLLAR_INDEX_WINDOW_DAYS}d",
        "real_yield_10y",
    }
    assert set(FEATURE_NAMES) == expected


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("macro", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "macro"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


def test_required_series_ids_include_eight_fred_series() -> None:
    """v1 needs exactly 8 FRED series — pin this so a future contributor
    can't silently drop a required series."""
    assert len(REQUIRED_SERIES_IDS) == 8
    assert set(REQUIRED_SERIES_IDS) == {
        FRED_TREASURY_10Y,
        FRED_TREASURY_2Y,
        FRED_TREASURY_3M,
        FRED_CORPORATE_BAA,
        FRED_CORPORATE_AAA,
        FRED_VIX,
        FRED_DOLLAR_INDEX,
        FRED_TIPS_10Y,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_zero_dollar_window() -> None:
    with pytest.raises(ValueError, match="dollar_index_window_days must be >= 1"):
        MacroConfig(dollar_index_window_days=0)


def test_default_config_uses_v1_defaults() -> None:
    assert DEFAULT_CONFIG.version == "macro-v1"
    assert DEFAULT_CONFIG.dollar_index_window_days == 30


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_aggregator_produces_one_row_per_date() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=10)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    panel = build_macro_panel(
        series_values=values,
        trading_dates=trading_dates,
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert len(panel.frame) == 10
    # All required series are columns in the output.
    for sid in REQUIRED_SERIES_IDS:
        assert sid in panel.frame.columns


def test_aggregator_forward_fills_holidays() -> None:
    """Macro feeds skip weekends and holidays; the aggregator
    forward-fills the most recent value to fill gaps."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=15)
    # Single observation; should propagate forward.
    values = [_series_value(FRED_TREASURY_10Y, _utc(2024, 6, 3), 4.50)]
    panel = build_macro_panel(
        series_values=values,
        trading_dates=trading_dates,
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    # All 15 trading days should carry the value 4.50.
    assert (panel.frame[FRED_TREASURY_10Y] == 4.50).all()
    # Other (non-observed) series are NaN.
    assert panel.frame[FRED_TREASURY_2Y].isna().all()


def test_aggregator_filters_unknown_series() -> None:
    """Records for series IDs outside REQUIRED_SERIES_IDS are silently
    ignored — operators can pass a superset of series."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = [
        _series_value(FRED_TREASURY_10Y, _utc(2024, 6, 3), 4.50),
        # Unknown series — should be ignored, not crash.
        _series_value("UNKNOWN_SERIES", _utc(2024, 6, 3), 99.9),
    ]
    panel = build_macro_panel(
        series_values=values,
        trading_dates=trading_dates,
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert "UNKNOWN_SERIES" not in panel.frame.columns
    assert (panel.frame[FRED_TREASURY_10Y] == 4.50).all()


def test_aggregator_empty_inputs_return_empty_data() -> None:
    """Empty trading_dates → empty frame regardless of input."""
    panel = build_macro_panel(
        series_values=[],
        trading_dates=pd.DatetimeIndex([]),
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert panel.frame.empty
    assert panel.n_observations_processed == 0


def test_aggregator_returns_typed_panel_object() -> None:
    panel = build_macro_panel(
        series_values=[],
        trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5),
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert isinstance(panel, AggregatedMacroPanel)


# ---------------------------------------------------------------------------
# Feature compute
# ---------------------------------------------------------------------------


def test_compute_broadcasts_across_instruments() -> None:
    """A single per-date macro panel produces N rows per date when
    broadcasted across N instruments."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    instruments = ("AAPL", "MSFT", "GOOG")
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values,
        instruments=instruments,
        trading_dates=trading_dates,
    )
    # 3 instruments × 5 trading days = 15 rows.
    assert len(ff.frame) == 15
    # And the same VIX value across all three instruments for any
    # given date.
    by_date = ff.frame.groupby("date")["vix_level"].nunique()
    assert (by_date == 1).all()  # one unique value per date (the same across instruments)


def test_compute_empty_inputs_return_empty_frame() -> None:
    ff = compute_macro_features(
        series_values=[],
        instruments=(),
        trading_dates=pd.DatetimeIndex([]),
    )
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_yield_curve_slope_matches_hand_computed() -> None:
    """DGS10 = 4.50, DGS2 = 4.80 → slope_10y_2y = -0.30 (inverted).
    DGS10 = 4.50, DGS3MO = 5.00 → slope_10y_3m = -0.50."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    row = ff.frame.iloc[0]
    assert row["yield_curve_slope_10y_2y"] == pytest.approx(-0.30)
    assert row["yield_curve_slope_10y_3m"] == pytest.approx(-0.50)


def test_credit_spread_matches_hand_computed() -> None:
    """BAA = 6.20, AAA = 5.40 → spread = 0.80."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    row = ff.frame.iloc[0]
    assert row["credit_spread_baa_aaa"] == pytest.approx(0.80)


def test_vix_level_direct_pass_through() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    assert ff.frame.iloc[0]["vix_level"] == pytest.approx(18.0)


def test_real_yield_10y_direct_pass_through() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    assert ff.frame.iloc[0]["real_yield_10y"] == pytest.approx(1.80)


def test_dollar_index_change_picks_up_30d_drift() -> None:
    """DXY = 100 at T, 110 at T+35 days → 30-day change = +10%."""
    trading_dates = _trading_dates(_utc(2024, 5, 1), n_days=120)
    values = [
        _series_value(FRED_DOLLAR_INDEX, _utc(2024, 5, 1), 100.0),
        _series_value(FRED_DOLLAR_INDEX, _utc(2024, 6, 5), 110.0),
    ]
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    change_col = f"dollar_index_change_{DEFAULT_DOLLAR_INDEX_WINDOW_DAYS}d"
    # In the transition window where current = 110 (after 2024-06-05)
    # and lagged = 100 (from 30 days before), the change is +10%.
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-06-20")].iloc[0]
    assert row[change_col] == pytest.approx(0.10)


def test_features_replace_infs_with_nan() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    for name in FEATURE_NAMES:
        column = ff.frame[name]
        assert not np.isinf(column.dropna()).any(), name


def test_compute_handles_missing_series_gracefully() -> None:
    """Missing required series produces NaN features instead of
    crashing — the family is resilient to partial feeds."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    # Only provide DGS10 — every other feature should be NaN.
    values = [_series_value(FRED_TREASURY_10Y, _utc(2024, 6, 3), 4.50)]
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    row = ff.frame.iloc[0]
    # yield_curve_slope_10y_2y needs DGS2 → NaN.
    assert pd.isna(row["yield_curve_slope_10y_2y"])
    assert pd.isna(row["credit_spread_baa_aaa"])
    assert pd.isna(row["vix_level"])
    assert pd.isna(row["real_yield_10y"])


def test_macro_values_identical_across_instruments_on_same_date() -> None:
    """The whole point of macro features is they're the same for every
    instrument on a given date."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    instruments = ("AAPL", "MSFT", "GOOG", "AMZN")
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=instruments, trading_dates=trading_dates
    )
    # For each date, all 4 instruments must have the same VIX (and
    # same of every other feature). nunique() returns 0 on
    # all-NaN cells (e.g. dollar_index_change_30d before the 30-day
    # window has filled) — that's fine, all-NaN across instruments
    # is still "consistent".
    for name in FEATURE_NAMES:
        unique_per_date = ff.frame.groupby("date")[name].nunique()
        assert (unique_per_date <= 1).all(), name


def test_compute_emits_six_features_per_row() -> None:
    """Sanity: the FeatureFrame's columns are exactly the 6 features
    plus the key columns."""
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    ff = compute_macro_features(
        series_values=values, instruments=("AAPL",), trading_dates=trading_dates
    )
    assert set(ff.frame.columns) == {"instrument_id", "date", *FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_aggregator_observations_processed_counts() -> None:
    values = _full_series_set(observation_date=_utc(2024, 6, 3))
    # Add one extra unknown-series observation; n_observations counts
    # ALL inputs, even those filtered out by the series-id check.
    extra = _series_value("UNKNOWN", _utc(2024, 6, 3), 1.0)
    panel = build_macro_panel(
        series_values=[*values, extra],
        trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5),
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert panel.n_observations_processed == 9  # 8 required + 1 unknown


def test_aggregator_handles_long_period_with_many_observations() -> None:
    """A year of daily observations stays panel-shaped and doesn't
    crash. Sanity check for the merge_asof per-series loop."""
    n_days = 252
    trading_dates = _trading_dates(_utc(2024, 1, 2), n_days=n_days)
    # One observation per day per required series.
    values: list[MacroSeriesValue] = []
    for sid in REQUIRED_SERIES_IDS:
        for i in range(n_days):
            obs_date = _utc(2024, 1, 2) + timedelta(days=i)
            values.append(_series_value(sid, obs_date, 1.0 + i * 0.001))
    panel = build_macro_panel(
        series_values=values,
        trading_dates=trading_dates,
        required_series_ids=REQUIRED_SERIES_IDS,
    )
    assert len(panel.frame) == n_days
    for sid in REQUIRED_SERIES_IDS:
        assert panel.frame[sid].notna().all(), sid
