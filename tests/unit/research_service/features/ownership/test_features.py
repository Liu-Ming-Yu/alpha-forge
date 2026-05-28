"""Unit tests for the ownership-v1 feature family."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.ownership import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MANIFEST,
    Holding13FRecord,
    OwnershipConfig,
    SharesOutstandingRecord,
    ShortInterestRecord,
    compute_ownership_features,
)
from quant_platform.research.features.ownership.aggregator import (
    AggregatedOwnershipPanel,
    build_ownership_panel,
)
from quant_platform.research.features.ownership.config import (
    DEFAULT_13F_CHANGE_WINDOW_DAYS,
    DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS,
    FEATURE_SET_VERSION,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _trading_dates(start: datetime, n_days: int) -> pd.DatetimeIndex:
    """Naive (tz-stripped) daily calendar, n_days business days from start."""
    naive_start = start.replace(tzinfo=None)
    return pd.date_range(naive_start, periods=n_days, freq="B")


def _synthetic_holdings(
    *,
    instrument_id: str = "AAPL",
    n_filers: int = 5,
    period_end: datetime,
    shares_per_filer: int = 1_000_000,
) -> list[Holding13FRecord]:
    return [
        Holding13FRecord(
            filer_id=f"FILER_{i}",
            instrument_id=instrument_id,
            period_end=period_end,
            shares_held=shares_per_filer,
            market_value=float(shares_per_filer * 150.0),
        )
        for i in range(n_filers)
    ]


def _synthetic_short_interest(
    *,
    instrument_id: str = "AAPL",
    settlement_date: datetime,
    short_shares: int = 500_000,
    avg_daily_volume: float = 1_000_000.0,
) -> ShortInterestRecord:
    return ShortInterestRecord(
        instrument_id=instrument_id,
        settlement_date=settlement_date,
        short_interest_shares=short_shares,
        avg_daily_volume_shares=avg_daily_volume,
    )


def _synthetic_shares_out(
    *,
    instrument_id: str = "AAPL",
    period_end: datetime,
    shares: int = 100_000_000,
) -> SharesOutstandingRecord:
    return SharesOutstandingRecord(
        instrument_id=instrument_id,
        period_end=period_end,
        shares_outstanding=shares,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_holding_13f_rejects_negative_shares() -> None:
    with pytest.raises(ValueError, match="shares_held must be >= 0"):
        Holding13FRecord(
            filer_id="F1",
            instrument_id="AAPL",
            period_end=_utc(2024, 3, 31),
            shares_held=-1,
            market_value=0.0,
        )


def test_holding_13f_rejects_naive_period_end() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Holding13FRecord(
            filer_id="F1",
            instrument_id="AAPL",
            period_end=datetime(2024, 3, 31),  # naive
            shares_held=1,
            market_value=0.0,
        )


def test_short_interest_rejects_zero_avg_volume() -> None:
    with pytest.raises(ValueError, match="avg_daily_volume_shares must be > 0"):
        ShortInterestRecord(
            instrument_id="AAPL",
            settlement_date=_utc(2024, 4, 15),
            short_interest_shares=100,
            avg_daily_volume_shares=0.0,
        )


def test_shares_outstanding_rejects_zero() -> None:
    with pytest.raises(ValueError, match="shares_outstanding must be > 0"):
        SharesOutstandingRecord(
            instrument_id="AAPL",
            period_end=_utc(2024, 3, 31),
            shares_outstanding=0,
        )


# ---------------------------------------------------------------------------
# Catalogue + manifest
# ---------------------------------------------------------------------------


def test_feature_specs_total_six() -> None:
    assert len(FEATURE_SPECS) == 6
    assert len(FEATURE_NAMES) == 6
    assert len(set(FEATURE_NAMES)) == 6


def test_feature_set_version_is_v1() -> None:
    assert FEATURE_SET_VERSION == "ownership-v1"


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "ownership"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {
        "institutional_ownership_pct",
        "institutional_holder_count",
        f"institutional_ownership_change_{DEFAULT_13F_CHANGE_WINDOW_DAYS}d",
        "short_interest_ratio",
        "days_to_cover",
        f"short_interest_change_{DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS}d",
    }
    assert set(FEATURE_NAMES) == expected


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("ownership", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "ownership"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_negative_availability_lag() -> None:
    with pytest.raises(ValueError, match="holding_13f_availability_lag_days must be >= 0"):
        OwnershipConfig(holding_13f_availability_lag_days=-1)


def test_config_rejects_zero_change_window() -> None:
    with pytest.raises(ValueError, match="holding_13f_change_window_days must be >= 1"):
        OwnershipConfig(holding_13f_change_window_days=0)


# ---------------------------------------------------------------------------
# Aggregator behaviour
# ---------------------------------------------------------------------------


def test_aggregator_produces_one_row_per_instrument_date() -> None:
    period_end = _utc(2024, 3, 31)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    panel = build_ownership_panel(
        holdings=_synthetic_holdings(period_end=period_end),
        short_interest=[],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    # 1 instrument × 10 trading days.
    assert len(panel.frame) == 10
    assert set(panel.frame["instrument_id"]) == {"AAPL"}
    # 5 filers × 1M shares = 5M institutional shares total.
    assert (panel.frame["institutional_shares_total"] == 5_000_000).all()
    assert (panel.frame["institutional_holder_count"] == 5).all()


def test_aggregator_masks_rows_before_availability() -> None:
    """13F rows shouldn't appear on the panel before their 45-day
    availability lag has elapsed."""
    period_end = _utc(2024, 3, 31)
    # Calendar straddles the availability boundary: 45 days after
    # 2024-03-31 is 2024-05-15.
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=60)
    panel = build_ownership_panel(
        holdings=_synthetic_holdings(period_end=period_end),
        short_interest=[],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    # Before 2024-05-15, institutional_shares_total must be NaN.
    early_rows = panel.frame[panel.frame["date"] < pd.Timestamp("2024-05-15")]
    assert early_rows["institutional_shares_total"].isna().all()
    # On / after the availability date, the values are populated.
    late_rows = panel.frame[panel.frame["date"] >= pd.Timestamp("2024-05-15")]
    assert late_rows["institutional_shares_total"].notna().all()


def test_aggregator_respects_explicit_available_at_override() -> None:
    """If the operator supplies ``available_at`` directly on a record,
    the aggregator uses it instead of the period_end + lag default."""
    period_end = _utc(2024, 3, 31)
    custom_available = _utc(2024, 4, 10)  # before the 45-day default
    holdings = [
        Holding13FRecord(
            filer_id="F1",
            instrument_id="AAPL",
            period_end=period_end,
            shares_held=1_000_000,
            market_value=150_000_000.0,
            available_at=custom_available,
        )
    ]
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=60)
    panel = build_ownership_panel(
        holdings=holdings,
        short_interest=[],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    # 2024-04-10 (custom_available) is BEFORE the default 2024-05-15;
    # the early-availability override should populate rows from then.
    row_at_april_15 = panel.frame[panel.frame["date"] == pd.Timestamp("2024-04-15")]
    assert len(row_at_april_15) == 1
    assert row_at_april_15.iloc[0]["institutional_shares_total"] == 1_000_000


def test_aggregator_empty_inputs_return_empty_frame() -> None:
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=10)
    panel = build_ownership_panel(
        holdings=[],
        short_interest=[],
        shares_outstanding=[],
        trading_dates=trading_dates,
    )
    assert panel.frame.empty
    assert panel.n_holdings_processed == 0
    assert panel.n_short_interest_processed == 0
    assert panel.n_shares_outstanding_processed == 0


def test_aggregator_forward_fills_through_dates_without_new_records() -> None:
    """The most recent available record should propagate to subsequent
    dates until a newer one arrives."""
    q1 = _utc(2024, 3, 31)
    q2 = _utc(2024, 6, 30)
    holdings = [
        *_synthetic_holdings(period_end=q1, n_filers=3),
        *_synthetic_holdings(period_end=q2, n_filers=5),
    ]
    shares_outs = [
        _synthetic_shares_out(period_end=q1),
        _synthetic_shares_out(period_end=q2),
    ]
    # Q1 available at 2024-05-15; Q2 available at 2024-08-14.
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=150)
    panel = build_ownership_panel(
        holdings=holdings,
        short_interest=[],
        shares_outstanding=shares_outs,
        trading_dates=trading_dates,
    )

    # On 2024-06-01: Q1 should be active, Q2 not yet → 3 holders.
    mid_june = panel.frame[panel.frame["date"] == pd.Timestamp("2024-06-03")]
    assert len(mid_june) == 1
    assert mid_june.iloc[0]["institutional_holder_count"] == 3
    # On 2024-09-01: Q2 is available, should override Q1 → 5 holders.
    september = panel.frame[panel.frame["date"] == pd.Timestamp("2024-09-02")]
    assert len(september) == 1
    assert september.iloc[0]["institutional_holder_count"] == 5


# ---------------------------------------------------------------------------
# Feature compute
# ---------------------------------------------------------------------------


def test_compute_features_shape_matches_trading_dates() -> None:
    period_end = _utc(2024, 3, 31)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=20)
    ff = compute_ownership_features(
        holdings=_synthetic_holdings(period_end=period_end),
        short_interest=[_synthetic_short_interest(settlement_date=_utc(2024, 5, 31))],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    assert len(ff.frame) == 20  # 1 instrument × 20 days
    for name in FEATURE_NAMES:
        assert name in ff.frame.columns


def test_compute_features_empty_inputs_return_empty_frame() -> None:
    ff = compute_ownership_features(
        holdings=[],
        short_interest=[],
        shares_outstanding=[],
        trading_dates=pd.DatetimeIndex([]),
    )
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_institutional_ownership_pct_matches_hand_computed() -> None:
    """5 filers × 1M shares = 5M total institutional shares.
    Shares outstanding = 100M → ownership_pct = 0.05."""
    period_end = _utc(2024, 3, 31)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    ff = compute_ownership_features(
        holdings=_synthetic_holdings(period_end=period_end, n_filers=5),
        short_interest=[],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    # Inspect a row after the 45-day availability lag.
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-06-03")]
    assert len(row) == 1
    assert row.iloc[0]["institutional_ownership_pct"] == pytest.approx(0.05)
    assert row.iloc[0]["institutional_holder_count"] == 5


def test_short_interest_ratio_and_days_to_cover_match_hand_computed() -> None:
    """short_interest = 500k, shares-out = 100M → ratio = 0.005.
    short_interest = 500k, avg_daily_vol = 1M → days_to_cover = 0.5."""
    period_end = _utc(2024, 3, 31)
    settlement = _utc(2024, 5, 15)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    ff = compute_ownership_features(
        holdings=[],
        short_interest=[
            _synthetic_short_interest(
                settlement_date=settlement,
                short_shares=500_000,
                avg_daily_volume=1_000_000.0,
            )
        ],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-06-03")]
    assert len(row) == 1
    assert row.iloc[0]["short_interest_ratio"] == pytest.approx(0.005)
    assert row.iloc[0]["days_to_cover"] == pytest.approx(0.5)


def test_institutional_ownership_change_picks_up_qoq_drift() -> None:
    """Q1: 3 filers × 1M shares. Q2: 5 filers × 1M shares. Shares-out
    constant. Then institutional_ownership_pct should rise from 0.03
    to 0.05 across the two filings — and the change feature should
    capture the +0.02 diff over the 63-day window."""
    q1 = _utc(2024, 3, 31)
    q2 = _utc(2024, 6, 30)
    holdings = [
        *_synthetic_holdings(period_end=q1, n_filers=3),
        *_synthetic_holdings(period_end=q2, n_filers=5),
    ]
    shares_outs = [
        _synthetic_shares_out(period_end=q1),
        _synthetic_shares_out(period_end=q2),
    ]
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=200)
    ff = compute_ownership_features(
        holdings=holdings,
        short_interest=[],
        shares_outstanding=shares_outs,
        trading_dates=trading_dates,
    )
    # The change feature equals +0.02 in the narrow window where
    # ``T`` is in the Q2-available era but ``T - 63 trading days`` is
    # still in the Q1-available era. The exact rows that satisfy this
    # depend on holiday calendars, so we assert the change feature
    # ATTAINS +0.02 somewhere in the panel rather than pinning a
    # specific row.
    change_col = f"institutional_ownership_change_{DEFAULT_13F_CHANGE_WINDOW_DAYS}d"
    nonzero_positive = ff.frame[change_col].dropna()
    nonzero_positive = nonzero_positive[nonzero_positive > 0]
    assert len(nonzero_positive) > 0, "no positive ownership change observed"
    # The peak diff should equal exactly +0.02 (Q1 pct 0.03 → Q2 pct 0.05).
    assert nonzero_positive.max() == pytest.approx(0.02)


def test_features_clipped_to_unit_interval() -> None:
    """institutional_ownership_pct and short_interest_ratio are clipped
    to [0, 1] even when vendor totals exceed shares-outstanding."""
    period_end = _utc(2024, 3, 31)
    settlement = _utc(2024, 5, 15)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    # Inject vendor inconsistency: 13F totals > shares-out.
    ff = compute_ownership_features(
        holdings=_synthetic_holdings(
            period_end=period_end, n_filers=10, shares_per_filer=20_000_000
        ),  # 200M institutional, 100M shares-out
        short_interest=[
            _synthetic_short_interest(
                settlement_date=settlement,
                short_shares=200_000_000,  # > shares-out
                avg_daily_volume=1_000_000.0,
            )
        ],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-06-03")].iloc[0]
    assert row["institutional_ownership_pct"] <= 1.0
    assert row["short_interest_ratio"] <= 1.0


def test_compute_handles_missing_shares_outstanding_gracefully() -> None:
    """Without shares-outstanding, the ratio features are NaN (safe_div
    on a NaN denominator) but the family doesn't crash and the other
    features (holder_count, days_to_cover) still compute."""
    period_end = _utc(2024, 3, 31)
    settlement = _utc(2024, 5, 15)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    ff = compute_ownership_features(
        holdings=_synthetic_holdings(period_end=period_end, n_filers=3),
        short_interest=[
            _synthetic_short_interest(
                settlement_date=settlement, short_shares=300_000, avg_daily_volume=1_000_000.0
            )
        ],
        shares_outstanding=[],  # no shares-outstanding records
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-06-03")].iloc[0]
    # Ratio features: NaN (no denominator).
    assert pd.isna(row["institutional_ownership_pct"])
    assert pd.isna(row["short_interest_ratio"])
    # holder_count + days_to_cover: still computable.
    assert row["institutional_holder_count"] == 3
    assert row["days_to_cover"] == pytest.approx(0.3)


def test_features_replace_infs_with_nan() -> None:
    """``safe_div`` already returns NaN on positive-only denominator,
    but the final ``.replace([inf,-inf], nan)`` is defence-in-depth."""
    period_end = _utc(2024, 3, 31)
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    ff = compute_ownership_features(
        holdings=_synthetic_holdings(period_end=period_end),
        short_interest=[],
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    for name in FEATURE_NAMES:
        column = ff.frame[name]
        assert not np.isinf(column.dropna()).any(), name


# ---------------------------------------------------------------------------
# Sanity / smoke
# ---------------------------------------------------------------------------


def test_panel_aggregates_match_processed_counts() -> None:
    period_end = _utc(2024, 3, 31)
    holdings = _synthetic_holdings(period_end=period_end, n_filers=4)
    si = [_synthetic_short_interest(settlement_date=_utc(2024, 5, 15))]
    so = [_synthetic_shares_out(period_end=period_end)]
    panel = build_ownership_panel(
        holdings=holdings,
        short_interest=si,
        shares_outstanding=so,
        trading_dates=_trading_dates(_utc(2024, 6, 1), n_days=5),
    )
    assert isinstance(panel, AggregatedOwnershipPanel)
    assert panel.n_holdings_processed == 4
    assert panel.n_short_interest_processed == 1
    assert panel.n_shares_outstanding_processed == 1


def test_default_config_uses_v1_defaults() -> None:
    assert DEFAULT_CONFIG.version == "ownership-v1"
    # 45-day 13F lag + 8-day FINRA lag are documented defaults.
    assert DEFAULT_CONFIG.holding_13f_availability_lag_days == 45
    assert DEFAULT_CONFIG.short_interest_availability_lag_days == 8


def test_multiple_instruments_keep_panels_independent() -> None:
    """Two instruments should produce two independent panel histories
    (no cross-contamination between AAPL and MSFT records)."""
    period_end = _utc(2024, 3, 31)
    holdings = [
        *_synthetic_holdings(instrument_id="AAPL", period_end=period_end, n_filers=3),
        *_synthetic_holdings(instrument_id="MSFT", period_end=period_end, n_filers=7),
    ]
    shares_outs = [
        _synthetic_shares_out(instrument_id="AAPL", period_end=period_end, shares=100_000_000),
        _synthetic_shares_out(instrument_id="MSFT", period_end=period_end, shares=200_000_000),
    ]
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=10)
    ff = compute_ownership_features(
        holdings=holdings,
        short_interest=[],
        shares_outstanding=shares_outs,
        trading_dates=trading_dates,
    )
    aapl = ff.frame[ff.frame["instrument_id"] == "AAPL"].iloc[0]
    msft = ff.frame[ff.frame["instrument_id"] == "MSFT"].iloc[0]
    assert aapl["institutional_holder_count"] == 3
    assert msft["institutional_holder_count"] == 7
    # AAPL: 3 filers × 1M / 100M = 0.03; MSFT: 7 × 1M / 200M = 0.035.
    assert aapl["institutional_ownership_pct"] == pytest.approx(0.03)
    assert msft["institutional_ownership_pct"] == pytest.approx(0.035)


def test_short_interest_change_picks_up_drift() -> None:
    """Two SI snapshots, one at 500k and one at 1.5M shares, with
    constant shares-out. The change feature should capture the +1%
    drift in short_interest_ratio."""
    period_end = _utc(2024, 3, 31)
    s1 = _utc(2024, 5, 15)
    # Settlement 30 calendar days later — well inside the 20-trading-day window.
    s2 = s1 + timedelta(days=45)
    short_interest = [
        _synthetic_short_interest(
            settlement_date=s1, short_shares=500_000, avg_daily_volume=1_000_000.0
        ),
        _synthetic_short_interest(
            settlement_date=s2, short_shares=1_500_000, avg_daily_volume=1_000_000.0
        ),
    ]
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=100)
    ff = compute_ownership_features(
        holdings=[],
        short_interest=short_interest,
        shares_outstanding=[_synthetic_shares_out(period_end=period_end)],
        trading_dates=trading_dates,
    )
    change_col = f"short_interest_change_{DEFAULT_SHORT_INTEREST_CHANGE_WINDOW_DAYS}d"
    # Same narrow-transition-window argument as the institutional
    # change test: assert the change feature ATTAINS +0.01 somewhere
    # in the panel rather than pinning a specific row.
    nonzero_positive = ff.frame[change_col].dropna()
    nonzero_positive = nonzero_positive[nonzero_positive > 0]
    assert len(nonzero_positive) > 0, "no positive short-interest change observed"
    # Peak diff = ratio_2 (0.015) − ratio_1 (0.005) = +0.01.
    assert nonzero_positive.max() == pytest.approx(0.01)
