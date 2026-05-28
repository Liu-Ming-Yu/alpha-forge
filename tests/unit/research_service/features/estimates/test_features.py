"""Unit tests for the estimates-v1 feature family."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.estimates import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MANIFEST,
    ConsensusSnapshot,
    EarningsSurpriseRecord,
    EstimatesConfig,
    compute_estimate_features,
)
from quant_platform.research.features.estimates.aggregator import (
    AggregatedEstimatesPanel,
    build_estimates_panel,
)
from quant_platform.research.features.estimates.config import (
    DEFAULT_REVISION_WINDOW_DAYS,
    DEFAULT_SURPRISE_LOOKBACK_QUARTERS,
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


def _consensus(
    *,
    instrument_id: str = "AAPL",
    snapshot_date: datetime,
    target_period: str = "FY1",
    estimate_kind: str = "eps",
    mean_estimate: float = 5.00,
    std_estimate: float | None = 0.20,
    n_estimates: int = 10,
    n_up_30d: int = 0,
    n_down_30d: int = 0,
) -> ConsensusSnapshot:
    return ConsensusSnapshot(
        instrument_id=instrument_id,
        snapshot_date=snapshot_date,
        target_period=target_period,
        estimate_kind=estimate_kind,
        mean_estimate=mean_estimate,
        std_estimate=std_estimate,
        n_estimates=n_estimates,
        n_up_revisions_30d=n_up_30d,
        n_down_revisions_30d=n_down_30d,
    )


def _surprise(
    *,
    instrument_id: str = "AAPL",
    fiscal_period_end: datetime,
    actual_eps: float = 1.10,
    consensus_mean_eps: float = 1.00,
    consensus_std_eps: float | None = 0.05,
    reported_at: datetime | None = None,
) -> EarningsSurpriseRecord:
    return EarningsSurpriseRecord(
        instrument_id=instrument_id,
        fiscal_period_end=fiscal_period_end,
        actual_eps=actual_eps,
        consensus_mean_eps=consensus_mean_eps,
        consensus_std_eps=consensus_std_eps,
        reported_at=reported_at or (fiscal_period_end + timedelta(days=30)),
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_consensus_rejects_zero_analysts() -> None:
    with pytest.raises(ValueError, match="n_estimates must be > 0"):
        ConsensusSnapshot(
            instrument_id="AAPL",
            snapshot_date=_utc(2024, 6, 1),
            target_period="FY1",
            estimate_kind="eps",
            mean_estimate=5.0,
            std_estimate=0.1,
            n_estimates=0,
        )


def test_consensus_rejects_invalid_target_period() -> None:
    with pytest.raises(ValueError, match="target_period must be one of"):
        ConsensusSnapshot(
            instrument_id="AAPL",
            snapshot_date=_utc(2024, 6, 1),
            target_period="FY99",  # invalid
            estimate_kind="eps",
            mean_estimate=5.0,
            std_estimate=0.1,
            n_estimates=10,
        )


def test_consensus_rejects_invalid_estimate_kind() -> None:
    with pytest.raises(ValueError, match="estimate_kind must be one of"):
        ConsensusSnapshot(
            instrument_id="AAPL",
            snapshot_date=_utc(2024, 6, 1),
            target_period="FY1",
            estimate_kind="ebitda",  # not yet supported
            mean_estimate=5.0,
            std_estimate=0.1,
            n_estimates=10,
        )


def test_consensus_allows_std_none_for_single_analyst() -> None:
    # Single-analyst coverage: std is undefined; None is the documented sentinel.
    c = ConsensusSnapshot(
        instrument_id="AAPL",
        snapshot_date=_utc(2024, 6, 1),
        target_period="FY1",
        estimate_kind="eps",
        mean_estimate=5.0,
        std_estimate=None,
        n_estimates=1,
    )
    assert c.std_estimate is None


def test_surprise_rejects_reported_before_period_end() -> None:
    with pytest.raises(ValueError, match="reported_at must be >="):
        EarningsSurpriseRecord(
            instrument_id="AAPL",
            fiscal_period_end=_utc(2024, 3, 31),
            actual_eps=1.0,
            consensus_mean_eps=1.0,
            consensus_std_eps=0.1,
            reported_at=_utc(2024, 3, 30),  # before period_end
        )


# ---------------------------------------------------------------------------
# Catalogue + manifest
# ---------------------------------------------------------------------------


def test_feature_specs_total_six() -> None:
    assert len(FEATURE_SPECS) == 6
    assert len(FEATURE_NAMES) == 6
    assert len(set(FEATURE_NAMES)) == 6


def test_feature_set_version_is_v1() -> None:
    assert FEATURE_SET_VERSION == "estimates-v1"


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "estimates"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {
        f"eps_estimate_revision_{DEFAULT_REVISION_WINDOW_DAYS}d",
        "eps_estimate_up_vs_down_30d",
        "eps_estimate_dispersion",
        "analyst_coverage_count",
        f"eps_surprise_mean_{DEFAULT_SURPRISE_LOOKBACK_QUARTERS}q",
        f"revenue_estimate_revision_{DEFAULT_REVISION_WINDOW_DAYS}d",
    }
    assert set(FEATURE_NAMES) == expected


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("estimates", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "estimates"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_invalid_target_period() -> None:
    with pytest.raises(ValueError, match="eps_target_period must be one of"):
        EstimatesConfig(eps_target_period="FY99")


def test_config_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="revision_window_days must be >= 1"):
        EstimatesConfig(revision_window_days=0)


def test_config_rejects_zero_surprise_lookback() -> None:
    with pytest.raises(ValueError, match="surprise_lookback_quarters must be >= 1"):
        EstimatesConfig(surprise_lookback_quarters=0)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_aggregator_filters_by_target_period_and_kind() -> None:
    """Snapshots for the wrong target_period or kind are silently
    dropped — they don't appear in the output frame's EPS / revenue
    columns."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [
        # FY1 EPS — should be picked up
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            target_period="FY1",
            estimate_kind="eps",
            mean_estimate=5.00,
        ),
        # FY2 EPS — should NOT be picked up (wrong target_period)
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            target_period="FY2",
            estimate_kind="eps",
            mean_estimate=6.00,
        ),
        # FY1 revenue — should be picked up for revenue column
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            target_period="FY1",
            estimate_kind="revenue",
            mean_estimate=400_000_000.0,
        ),
    ]
    panel = build_estimates_panel(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
        eps_target_period="FY1",
        revenue_target_period="FY1",
        revision_window_days=30,
        surprise_lookback_quarters=4,
    )
    inspect = panel.frame[panel.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    # FY1 EPS came through.
    assert inspect["eps_mean"] == 5.00
    # FY1 revenue came through.
    assert inspect["revenue_mean"] == 400_000_000.0
    # FY2 EPS did NOT contaminate — the FY1 EPS mean is 5.00, not 6.00.


def test_aggregator_forward_fills_consensus_through_quiet_days() -> None:
    """The most-recent snapshot should propagate forward until a newer
    one arrives."""
    trading_dates = _trading_dates(_utc(2024, 6, 1), n_days=30)
    snapshots = [
        _consensus(snapshot_date=_utc(2024, 6, 3), mean_estimate=5.00),
        _consensus(snapshot_date=_utc(2024, 6, 20), mean_estimate=5.20),
    ]
    panel = build_estimates_panel(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
        eps_target_period="FY1",
        revenue_target_period="FY1",
        revision_window_days=30,
        surprise_lookback_quarters=4,
    )
    # 2024-06-10: only the first snapshot is available → mean = 5.00.
    early = panel.frame[panel.frame["date"] == pd.Timestamp("2024-06-10")].iloc[0]
    assert early["eps_mean"] == 5.00
    # 2024-06-25: the second snapshot is now available → mean = 5.20.
    late = panel.frame[panel.frame["date"] == pd.Timestamp("2024-06-25")].iloc[0]
    assert late["eps_mean"] == 5.20


def test_aggregator_masks_surprise_records_before_reported_at() -> None:
    trading_dates = _trading_dates(_utc(2024, 4, 1), n_days=120)
    surprise = _surprise(
        instrument_id="AAPL",
        fiscal_period_end=_utc(2024, 3, 31),
        actual_eps=1.10,
        consensus_mean_eps=1.00,
        reported_at=_utc(2024, 5, 1),
    )
    panel = build_estimates_panel(
        consensus_snapshots=[],
        surprise_records=[surprise],
        trading_dates=trading_dates,
        eps_target_period="FY1",
        revenue_target_period="FY1",
        revision_window_days=30,
        surprise_lookback_quarters=4,
    )
    # Before 2024-05-01: surprise feature must be NaN.
    early = panel.frame[panel.frame["date"] < pd.Timestamp("2024-05-01")]
    assert early["eps_surprise_mean_recent"].isna().all()
    # On / after 2024-05-01: populated with 10% surprise.
    late = panel.frame[panel.frame["date"] >= pd.Timestamp("2024-05-01")]
    assert np.allclose(late["eps_surprise_mean_recent"].to_numpy(), 0.10)


def test_aggregator_empty_inputs_return_empty_frame() -> None:
    panel = build_estimates_panel(
        consensus_snapshots=[],
        surprise_records=[],
        trading_dates=_trading_dates(_utc(2024, 7, 1), n_days=10),
        eps_target_period="FY1",
        revenue_target_period="FY1",
        revision_window_days=30,
        surprise_lookback_quarters=4,
    )
    assert panel.frame.empty
    assert panel.n_consensus_processed == 0
    assert panel.n_surprise_processed == 0


# ---------------------------------------------------------------------------
# Feature compute
# ---------------------------------------------------------------------------


def test_compute_features_shape_matches_trading_dates() -> None:
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=20)
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15))]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    assert len(ff.frame) == 20
    for name in FEATURE_NAMES:
        assert name in ff.frame.columns


def test_compute_features_empty_inputs_return_empty_frame() -> None:
    ff = compute_estimate_features(
        consensus_snapshots=[],
        surprise_records=[],
        trading_dates=pd.DatetimeIndex([]),
    )
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_eps_estimate_revision_picks_up_30d_drift() -> None:
    """Two snapshots: FY1 EPS at 5.00 on 2024-06-15, 5.25 on 2024-07-20.
    The revision feature equals +5% exactly in the window where the
    current consensus is the 2nd snapshot (5.25) and the lagged
    consensus (most-recent snapshot whose date ≤ panel_date − 30
    days) is still the 1st snapshot (5.00). That window is when
    panel_date ∈ [2024-07-20, 2024-08-19] (= snapshot_2.date through
    snapshot_2.date + 30 days). Outside that window the revision is
    either 0 (both pointers on snapshot 2) or NaN (no lagged).
    """
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=80)
    snapshots = [
        _consensus(snapshot_date=_utc(2024, 6, 15), mean_estimate=5.00),
        _consensus(snapshot_date=_utc(2024, 7, 20), mean_estimate=5.25),
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    revision_col = f"eps_estimate_revision_{DEFAULT_REVISION_WINDOW_DAYS}d"
    # Inspect inside the transition window: 2024-08-01 is past
    # snapshot 2 (so current = 5.25) but 30 days back is 2024-07-02,
    # which is still snapshot 1's era (so lagged = 5.00).
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-08-01")].iloc[0]
    assert row[revision_col] == pytest.approx(0.05)


def test_eps_up_vs_down_matches_hand_computed() -> None:
    """6 up - 2 down = +4 net out of 8 total → ratio = 0.5."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            n_up_30d=6,
            n_down_30d=2,
        )
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert row["eps_estimate_up_vs_down_30d"] == pytest.approx((6 - 2) / (6 + 2))


def test_eps_dispersion_matches_hand_computed() -> None:
    """mean=5.00, std=0.20 → dispersion = 0.20/5.00 = 0.04."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15), mean_estimate=5.00, std_estimate=0.20)]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert row["eps_estimate_dispersion"] == pytest.approx(0.04)


def test_eps_dispersion_is_nan_for_single_analyst() -> None:
    """std=None → dispersion = NaN."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            mean_estimate=5.00,
            std_estimate=None,
            n_estimates=1,
        )
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert pd.isna(row["eps_estimate_dispersion"])


def test_analyst_coverage_count_matches_n_estimates() -> None:
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15), n_estimates=14)]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert row["analyst_coverage_count"] == 14.0


def test_eps_surprise_mean_averages_last_four_quarters() -> None:
    """Four consecutive quarters with surprises +10%, -5%, +15%, +20%.
    Mean of the last 4 = (10 - 5 + 15 + 20)/4 = 10% = 0.10."""
    trading_dates = _trading_dates(_utc(2025, 4, 1), n_days=20)
    surprises = [
        _surprise(
            fiscal_period_end=_utc(2024, 3, 31),
            actual_eps=1.10,
            consensus_mean_eps=1.00,
            reported_at=_utc(2024, 5, 1),
        ),
        _surprise(
            fiscal_period_end=_utc(2024, 6, 30),
            actual_eps=0.95,
            consensus_mean_eps=1.00,
            reported_at=_utc(2024, 7, 30),
        ),
        _surprise(
            fiscal_period_end=_utc(2024, 9, 30),
            actual_eps=1.15,
            consensus_mean_eps=1.00,
            reported_at=_utc(2024, 10, 30),
        ),
        _surprise(
            fiscal_period_end=_utc(2024, 12, 31),
            actual_eps=1.20,
            consensus_mean_eps=1.00,
            reported_at=_utc(2025, 1, 30),
        ),
    ]
    ff = compute_estimate_features(
        consensus_snapshots=[],
        surprise_records=surprises,
        trading_dates=trading_dates,
    )
    surprise_col = f"eps_surprise_mean_{DEFAULT_SURPRISE_LOOKBACK_QUARTERS}q"
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2025-04-02")].iloc[0]
    # (0.10 + -0.05 + 0.15 + 0.20) / 4 = 0.10.
    assert row[surprise_col] == pytest.approx(0.10)


def test_revenue_estimate_revision_picks_up_drift() -> None:
    """Same logic as the EPS test, but for revenue. Inspect the same
    transition window."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=80)
    snapshots = [
        _consensus(
            snapshot_date=_utc(2024, 6, 15),
            estimate_kind="revenue",
            mean_estimate=400_000_000.0,
        ),
        _consensus(
            snapshot_date=_utc(2024, 7, 20),
            estimate_kind="revenue",
            mean_estimate=420_000_000.0,
        ),
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    revision_col = f"revenue_estimate_revision_{DEFAULT_REVISION_WINDOW_DAYS}d"
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-08-01")].iloc[0]
    # (420M - 400M) / 400M = 0.05.
    assert row[revision_col] == pytest.approx(0.05)


def test_up_vs_down_lives_in_minus_one_to_one() -> None:
    """The ratio is bounded in [-1, 1] by construction."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    # All upward revisions: 5 up, 0 down → ratio = 1.0.
    snapshots_all_up = [_consensus(snapshot_date=_utc(2024, 6, 15), n_up_30d=5, n_down_30d=0)]
    ff_up = compute_estimate_features(
        consensus_snapshots=snapshots_all_up,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row_up = ff_up.frame[ff_up.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert row_up["eps_estimate_up_vs_down_30d"] == pytest.approx(1.0)
    # All downward: 0 up, 5 down → -1.0.
    snapshots_all_down = [_consensus(snapshot_date=_utc(2024, 6, 15), n_up_30d=0, n_down_30d=5)]
    ff_down = compute_estimate_features(
        consensus_snapshots=snapshots_all_down,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row_down = ff_down.frame[ff_down.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert row_down["eps_estimate_up_vs_down_30d"] == pytest.approx(-1.0)


def test_up_vs_down_is_nan_when_no_revisions() -> None:
    """No analysts revised → ratio is undefined (0/0) → NaN."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15), n_up_30d=0, n_down_30d=0)]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-07-02")].iloc[0]
    assert pd.isna(row["eps_estimate_up_vs_down_30d"])


def test_revision_nan_when_lagged_consensus_zero() -> None:
    """Lagged consensus = 0 → safe_div produces NaN, not inf.

    Use the same transition-window argument as the drift test:
    inspect inside [2024-07-20, 2024-08-19] where current=0.5
    (snapshot 2) but lagged=0.0 (snapshot 1)."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=80)
    snapshots = [
        _consensus(snapshot_date=_utc(2024, 6, 15), mean_estimate=0.0),
        _consensus(snapshot_date=_utc(2024, 7, 20), mean_estimate=0.5),
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    revision_col = f"eps_estimate_revision_{DEFAULT_REVISION_WINDOW_DAYS}d"
    row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-08-01")].iloc[0]
    assert pd.isna(row[revision_col])


def test_features_replace_infs_with_nan() -> None:
    """``safe_div`` + the boundary replace([inf,-inf], nan) protects
    the output from infinities."""
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15))]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    for name in FEATURE_NAMES:
        column = ff.frame[name]
        assert not np.isinf(column.dropna()).any(), name


def test_multiple_instruments_keep_panels_independent() -> None:
    trading_dates = _trading_dates(_utc(2024, 7, 1), n_days=10)
    snapshots = [
        _consensus(instrument_id="AAPL", snapshot_date=_utc(2024, 6, 15), n_estimates=12),
        _consensus(instrument_id="MSFT", snapshot_date=_utc(2024, 6, 15), n_estimates=20),
    ]
    ff = compute_estimate_features(
        consensus_snapshots=snapshots,
        surprise_records=[],
        trading_dates=trading_dates,
    )
    aapl = ff.frame[ff.frame["instrument_id"] == "AAPL"].iloc[0]
    msft = ff.frame[ff.frame["instrument_id"] == "MSFT"].iloc[0]
    assert aapl["analyst_coverage_count"] == 12
    assert msft["analyst_coverage_count"] == 20


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_panel_processed_counts() -> None:
    snapshots = [_consensus(snapshot_date=_utc(2024, 6, 15)) for _ in range(5)]
    surprises = [
        _surprise(fiscal_period_end=_utc(2024, 3, 31)),
        _surprise(fiscal_period_end=_utc(2024, 6, 30)),
    ]
    panel = build_estimates_panel(
        consensus_snapshots=snapshots,
        surprise_records=surprises,
        trading_dates=_trading_dates(_utc(2024, 7, 1), n_days=10),
        eps_target_period="FY1",
        revenue_target_period="FY1",
        revision_window_days=30,
        surprise_lookback_quarters=4,
    )
    assert isinstance(panel, AggregatedEstimatesPanel)
    assert panel.n_consensus_processed == 5
    assert panel.n_surprise_processed == 2


def test_default_config_uses_v1_defaults() -> None:
    assert DEFAULT_CONFIG.version == "estimates-v1"
    assert DEFAULT_CONFIG.eps_target_period == "FY1"
    assert DEFAULT_CONFIG.revenue_target_period == "FY1"
    assert DEFAULT_CONFIG.revision_window_days == 30
    assert DEFAULT_CONFIG.surprise_lookback_quarters == 4
