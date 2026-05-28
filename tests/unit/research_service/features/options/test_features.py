"""Unit tests for the options-v1 feature family."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.options import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MANIFEST,
    OptionsConfig,
    OptionsSnapshot,
    compute_options_features,
)
from quant_platform.research.features.options.aggregator import (
    AggregatedOptionsPanel,
    build_options_panel,
)
from quant_platform.research.features.options.config import (
    DEFAULT_ATM_TENOR_DAYS,
    DEFAULT_REALIZED_VOL_WINDOW_DAYS,
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


def _snapshot(
    *,
    instrument_id: str = "AAPL",
    snapshot_date: datetime,
    iv_30d_atm: float | None = 0.25,
    iv_60d_atm: float | None = 0.27,
    iv_25d_call: float | None = 0.24,
    iv_25d_put: float | None = 0.28,
    put_volume: int = 1_000_000,
    call_volume: int = 800_000,
    put_open_interest: int = 5_000_000,
    call_open_interest: int = 6_000_000,
    realized_vol_21d: float | None = 0.20,
) -> OptionsSnapshot:
    return OptionsSnapshot(
        instrument_id=instrument_id,
        snapshot_date=snapshot_date,
        iv_30d_atm=iv_30d_atm,
        iv_60d_atm=iv_60d_atm,
        iv_25d_call=iv_25d_call,
        iv_25d_put=iv_25d_put,
        put_volume=put_volume,
        call_volume=call_volume,
        put_open_interest=put_open_interest,
        call_open_interest=call_open_interest,
        realized_vol_21d=realized_vol_21d,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_snapshot_rejects_negative_iv() -> None:
    with pytest.raises(ValueError, match="iv_30d_atm must be >= 0"):
        OptionsSnapshot(
            instrument_id="AAPL",
            snapshot_date=_utc(2024, 6, 1),
            iv_30d_atm=-0.1,
            iv_60d_atm=0.27,
            iv_25d_call=0.24,
            iv_25d_put=0.28,
            put_volume=100,
            call_volume=100,
            put_open_interest=100,
            call_open_interest=100,
            realized_vol_21d=0.20,
        )


def test_snapshot_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="put_volume must be >= 0"):
        OptionsSnapshot(
            instrument_id="AAPL",
            snapshot_date=_utc(2024, 6, 1),
            iv_30d_atm=0.25,
            iv_60d_atm=0.27,
            iv_25d_call=0.24,
            iv_25d_put=0.28,
            put_volume=-1,
            call_volume=100,
            put_open_interest=100,
            call_open_interest=100,
            realized_vol_21d=0.20,
        )


def test_snapshot_rejects_naive_date() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        OptionsSnapshot(
            instrument_id="AAPL",
            snapshot_date=datetime(2024, 6, 1),  # naive
            iv_30d_atm=0.25,
            iv_60d_atm=0.27,
            iv_25d_call=0.24,
            iv_25d_put=0.28,
            put_volume=100,
            call_volume=100,
            put_open_interest=100,
            call_open_interest=100,
            realized_vol_21d=0.20,
        )


def test_snapshot_allows_none_iv_for_illiquid_names() -> None:
    s = OptionsSnapshot(
        instrument_id="AAPL",
        snapshot_date=_utc(2024, 6, 1),
        iv_30d_atm=None,
        iv_60d_atm=None,
        iv_25d_call=None,
        iv_25d_put=None,
        put_volume=0,
        call_volume=0,
        put_open_interest=0,
        call_open_interest=0,
        realized_vol_21d=None,
    )
    assert s.iv_30d_atm is None


# ---------------------------------------------------------------------------
# Catalogue + manifest
# ---------------------------------------------------------------------------


def test_feature_specs_total_six() -> None:
    assert len(FEATURE_SPECS) == 6
    assert len(FEATURE_NAMES) == 6
    assert len(set(FEATURE_NAMES)) == 6


def test_feature_set_version_is_v1() -> None:
    assert FEATURE_SET_VERSION == "options-v1"


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "options"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {
        f"iv_{DEFAULT_ATM_TENOR_DAYS}d_atm",
        "iv_skew_25d",
        "iv_term_slope",
        "put_call_volume_ratio",
        "put_call_oi_ratio",
        f"iv_realized_premium_{DEFAULT_ATM_TENOR_DAYS}d",
    }
    assert set(FEATURE_NAMES) == expected


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("options", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "options"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_long_tenor_not_above_short() -> None:
    with pytest.raises(ValueError, match="term_long_tenor_days must be strictly greater"):
        OptionsConfig(atm_tenor_days=30, term_long_tenor_days=30)
    with pytest.raises(ValueError, match="term_long_tenor_days must be strictly greater"):
        OptionsConfig(atm_tenor_days=30, term_long_tenor_days=20)


def test_config_rejects_zero_realized_window() -> None:
    with pytest.raises(ValueError, match="realized_vol_window_days must be >= 2"):
        OptionsConfig(realized_vol_window_days=1)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_aggregator_forward_fills_snapshots() -> None:
    """Latest snapshot's values forward-fill onto subsequent days
    until a newer snapshot arrives."""
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=30)
    snapshots = [
        _snapshot(snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.20),
        _snapshot(snapshot_date=_utc(2024, 6, 17), iv_30d_atm=0.30),
    ]
    panel = build_options_panel(snapshots=snapshots, trading_dates=trading_dates)
    early = panel.frame[panel.frame["date"] == pd.Timestamp("2024-06-10")].iloc[0]
    late = panel.frame[panel.frame["date"] == pd.Timestamp("2024-06-25")].iloc[0]
    assert early["iv_30d_atm"] == 0.20
    assert late["iv_30d_atm"] == 0.30


def test_aggregator_empty_inputs_return_empty_frame() -> None:
    panel = build_options_panel(
        snapshots=[], trading_dates=_trading_dates(_utc(2024, 6, 1), n_days=10)
    )
    assert panel.frame.empty
    assert panel.n_snapshots_processed == 0


def test_aggregator_returns_typed_panel_object() -> None:
    snap = _snapshot(snapshot_date=_utc(2024, 6, 3))
    panel = build_options_panel(
        snapshots=[snap], trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=10)
    )
    assert isinstance(panel, AggregatedOptionsPanel)
    assert panel.n_snapshots_processed == 1


# ---------------------------------------------------------------------------
# Feature compute
# ---------------------------------------------------------------------------


def test_compute_shape_matches_trading_dates() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=20)
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3))]
    ff = compute_options_features(snapshots=snapshots, trading_dates=trading_dates)
    assert len(ff.frame) == 20
    for name in FEATURE_NAMES:
        assert name in ff.frame.columns


def test_compute_empty_inputs_return_empty_frame() -> None:
    ff = compute_options_features(snapshots=[], trading_dates=pd.DatetimeIndex([]))
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_iv_30d_atm_direct_pass_through() -> None:
    """iv_30d_atm feature = raw input value."""
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.25)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    row = ff.frame.iloc[0]
    assert row[f"iv_{DEFAULT_ATM_TENOR_DAYS}d_atm"] == pytest.approx(0.25)


def test_iv_skew_matches_put_minus_call() -> None:
    """skew = iv_25d_put - iv_25d_call = 0.28 - 0.24 = 0.04."""
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_25d_put=0.28, iv_25d_call=0.24)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    row = ff.frame.iloc[0]
    assert row["iv_skew_25d"] == pytest.approx(0.04)


def test_iv_skew_nan_when_either_leg_missing() -> None:
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_25d_put=None, iv_25d_call=0.24)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert pd.isna(ff.frame.iloc[0]["iv_skew_25d"])


def test_iv_term_slope_matches_hand_computed() -> None:
    """slope = (iv_60d - iv_30d) / iv_30d = (0.27 - 0.25) / 0.25 = 0.08."""
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.25, iv_60d_atm=0.27)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    row = ff.frame.iloc[0]
    assert row["iv_term_slope"] == pytest.approx(0.08)


def test_iv_term_slope_nan_when_short_iv_zero() -> None:
    """Zero short-tenor IV → safe_div returns NaN, not inf."""
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.0, iv_60d_atm=0.27)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert pd.isna(ff.frame.iloc[0]["iv_term_slope"])


def test_put_call_volume_ratio_matches_hand_computed() -> None:
    """ratio = put_volume / call_volume = 1M / 800K = 1.25."""
    snapshots = [
        _snapshot(snapshot_date=_utc(2024, 6, 3), put_volume=1_000_000, call_volume=800_000)
    ]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert ff.frame.iloc[0]["put_call_volume_ratio"] == pytest.approx(1.25)


def test_put_call_volume_ratio_nan_on_zero_call_volume() -> None:
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), put_volume=100, call_volume=0)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert pd.isna(ff.frame.iloc[0]["put_call_volume_ratio"])


def test_put_call_oi_ratio_matches_hand_computed() -> None:
    """ratio = put_oi / call_oi = 5M / 6M ≈ 0.833."""
    snapshots = [
        _snapshot(
            snapshot_date=_utc(2024, 6, 3),
            put_open_interest=5_000_000,
            call_open_interest=6_000_000,
        )
    ]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert ff.frame.iloc[0]["put_call_oi_ratio"] == pytest.approx(5 / 6)


def test_iv_realized_premium_matches_hand_computed() -> None:
    """premium = iv_30d - realized_21d = 0.25 - 0.20 = 0.05."""
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.25, realized_vol_21d=0.20)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert ff.frame.iloc[0][f"iv_realized_premium_{DEFAULT_ATM_TENOR_DAYS}d"] == pytest.approx(0.05)


def test_iv_realized_premium_nan_when_realized_missing() -> None:
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3), realized_vol_21d=None)]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    assert pd.isna(ff.frame.iloc[0][f"iv_realized_premium_{DEFAULT_ATM_TENOR_DAYS}d"])


def test_features_replace_infs_with_nan() -> None:
    snapshots = [_snapshot(snapshot_date=_utc(2024, 6, 3))]
    ff = compute_options_features(
        snapshots=snapshots, trading_dates=_trading_dates(_utc(2024, 6, 5), n_days=5)
    )
    for name in FEATURE_NAMES:
        column = ff.frame[name]
        assert not np.isinf(column.dropna()).any(), name


def test_multiple_instruments_keep_panels_independent() -> None:
    trading_dates = _trading_dates(_utc(2024, 6, 5), n_days=5)
    snapshots = [
        _snapshot(instrument_id="AAPL", snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.25),
        _snapshot(instrument_id="MSFT", snapshot_date=_utc(2024, 6, 3), iv_30d_atm=0.35),
    ]
    ff = compute_options_features(snapshots=snapshots, trading_dates=trading_dates)
    aapl = ff.frame[ff.frame["instrument_id"] == "AAPL"].iloc[0]
    msft = ff.frame[ff.frame["instrument_id"] == "MSFT"].iloc[0]
    assert aapl[f"iv_{DEFAULT_ATM_TENOR_DAYS}d_atm"] == pytest.approx(0.25)
    assert msft[f"iv_{DEFAULT_ATM_TENOR_DAYS}d_atm"] == pytest.approx(0.35)


def test_default_config_uses_v1_defaults() -> None:
    assert DEFAULT_CONFIG.version == "options-v1"
    assert DEFAULT_CONFIG.atm_tenor_days == 30
    assert DEFAULT_CONFIG.term_long_tenor_days == 60
    assert DEFAULT_CONFIG.realized_vol_window_days == DEFAULT_REALIZED_VOL_WINDOW_DAYS
