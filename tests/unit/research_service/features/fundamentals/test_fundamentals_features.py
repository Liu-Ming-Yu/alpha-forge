"""Unit tests for ``fundamentals-plus-v1``.

Exact formulas are verified on hand-built synthetic Sharadar panels.
Each test pins one acceptance criterion from the Phase 2 brief:

1. Every produced column has a registered FeatureSpec.
2. Legacy 9 features remain positive-oriented in the spec catalog.
3. cfo_to_net_income is an alias of cash_conversion (canonical_name).
4. TTM aggregates honour the full 4-quarter window (warm-up NaN).
5. YoY ratios honour the full 5-quarter requirement (warm-up NaN).
6. Acceleration features are YoY deltas of the underlying growth /
   margin / quality columns.
7. Negative-premium quantities exit as ``low_*`` and are sign-flipped.
8. No cross-instrument leakage.
9. Sector neutralization subtracts per-sector medians via the shared
   ``neutralize_by_group`` helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features import get_global_registry
from quant_platform.research.features.fundamentals import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FundamentalsConfig,
    compute_fundamentals_features,
)
from quant_platform.research.features.neutralization import neutralize_feature_frame
from quant_platform.research.fundamentals.sharadar import SharadarPanel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(instrument_id: str, ticker: str, idx: int, **overrides: object) -> dict:
    """One quarterly row with deterministic, easy-to-reason-about values."""
    datekey = pd.Timestamp("2022-01-31") + pd.Timedelta(days=90 * idx)
    base = {
        "instrument_id": instrument_id,
        "ticker": ticker,
        "datekey": datekey,
        "calendardate": datekey,
        "revenue": 100.0,
        "cor": 60.0,
        "gp": 40.0,
        "opex": 25.0,
        "netinc": 10.0,
        "assets": 500.0,
        "liabilities": 200.0,
        "equity": 300.0,
        "debt": 100.0,
        "cashneq": 50.0,
        "ncfo": 12.0,
        "fcf": 8.0,
        "capex": -4.0,
        "marketcap": 1_000.0,
        "pe": 10.0,
        "pb": 2.0,
        "ps": 1.5,
        "divyield": 0.02,
        "sharesbas": 1_000_000,
    }
    base.update(overrides)
    return base


def _build_panel(rows: list[dict]) -> SharadarPanel:
    frame = pd.DataFrame(rows)
    return SharadarPanel(
        frame=frame,
        instrument_coverage=int(frame["instrument_id"].nunique()),
        datekey_min=frame["datekey"].min(),
        datekey_max=frame["datekey"].max(),
        dropped_no_instrument_id=(),
        dropped_missing_datekey=0,
        duplicates_resolved=0,
    )


@pytest.fixture
def steady_panel() -> SharadarPanel:
    """One instrument, 8 quarters of identical inputs. Lets exact-value
    assertions on per-quarter ratios collapse to single scalars."""
    return _build_panel(
        [_row("instrument-a", "AAA", idx) for idx in range(8)],
    )


@pytest.fixture
def two_instrument_panel() -> SharadarPanel:
    """Two instruments with different scales but identical histories."""
    rows = []
    for idx in range(8):
        rows.append(_row("instrument-a", "AAA", idx))
        rows.append(
            _row(
                "instrument-b",
                "BBB",
                idx,
                revenue=200.0,
                gp=80.0,
                opex=50.0,
                netinc=20.0,
                assets=1_000.0,
                equity=600.0,
                debt=200.0,
                cashneq=100.0,
                ncfo=24.0,
                fcf=16.0,
                marketcap=2_000.0,
            )
        )
    return _build_panel(rows)


# ---------------------------------------------------------------------------
# Spec / registry contract
# ---------------------------------------------------------------------------


def test_every_produced_column_has_a_spec(steady_panel: SharadarPanel) -> None:
    result = compute_fundamentals_features(steady_panel)
    assert set(result.feature_names) == set(FEATURE_NAMES)
    for name in result.feature_names:
        spec = result.feature_specs[name]
        assert spec.name == name
        assert spec.family == "fundamentals"
        assert spec.point_in_time is True
        assert spec.version == "fundamentals-plus-v1"


def test_feature_specs_registered_globally() -> None:
    registry = get_global_registry()
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name)
        assert registry.get(spec.name, spec.version) == spec


def test_legacy_nine_remain_positive_oriented() -> None:
    legacy_names = {
        "roe_ttm",
        "gross_profitability_q",
        "low_accruals_4q",
        "low_asset_growth_yoy",
        "fcf_yield_ttm",
        "cash_to_assets",
        "low_debt_to_equity",
        "book_to_price",
        "earnings_to_price",
    }
    spec_by_name = {spec.name: spec for spec in FEATURE_SPECS}
    for name in legacy_names:
        spec = spec_by_name[name]
        assert spec.expected_direction == "+", name
        assert spec.larger_is_better is True, name


def test_cfo_to_net_income_is_alias_of_cash_conversion() -> None:
    spec_by_name = {spec.name: spec for spec in FEATURE_SPECS}
    alias = spec_by_name["cfo_to_net_income"]
    canonical = spec_by_name["cash_conversion"]
    # Bidirectional alias rule (see FeatureSpec.aliases docstring):
    # the alias points at the canonical via canonical_name and carries
    # an empty aliases tuple; the canonical lists its alternate names
    # in its aliases tuple.
    assert alias.canonical_name == "cash_conversion"
    assert alias.aliases == ()
    assert canonical.canonical_name is None
    assert "cfo_to_net_income" in canonical.aliases
    assert alias.is_alias is True
    assert canonical.is_alias is False


def test_training_feature_names_drop_aliases(steady_panel: SharadarPanel) -> None:
    result = compute_fundamentals_features(steady_panel)
    training = set(result.training_feature_names)
    assert "cash_conversion" in training
    assert "cfo_to_net_income" not in training


# ---------------------------------------------------------------------------
# Exact formulas
# ---------------------------------------------------------------------------


def _last(result: pd.DataFrame, instrument: str = "instrument-a") -> pd.Series:
    return result.loc[result["instrument_id"] == instrument].sort_values("datekey").iloc[-1]


def test_quality_block_exact_values(steady_panel: SharadarPanel) -> None:
    result = compute_fundamentals_features(steady_panel).frame
    last = _last(result)

    # On the steady panel: revenue=100, gp=40, opex=25, netinc=10, assets=500,
    # equity=300, debt=100, cashneq=50. opinc = gp - opex = 15.
    assert last["gross_margin"] == pytest.approx(40.0 / 100.0)
    assert last["operating_margin"] == pytest.approx(15.0 / 100.0)
    assert last["net_margin"] == pytest.approx(10.0 / 100.0)
    assert last["asset_turnover"] == pytest.approx(100.0 / 500.0)
    assert last["equity_multiplier"] == pytest.approx(500.0 / 300.0)
    assert last["gross_profitability_q"] == pytest.approx(40.0 / 500.0)
    assert last["cash_to_assets"] == pytest.approx(50.0 / 500.0)

    # netinc_ttm = 40, equity_4q_avg = 300, assets_4q_avg = 500.
    assert last["roe_ttm"] == pytest.approx(40.0 / 300.0)
    assert last["roa_ttm"] == pytest.approx(40.0 / 500.0)
    # roic = netinc_ttm / (debt + equity - cashneq) = 40 / (100+300-50) = 40/350.
    assert last["roic_ttm"] == pytest.approx(40.0 / 350.0)


def test_growth_block_is_zero_on_steady_panel(steady_panel: SharadarPanel) -> None:
    """If revenue, gp, opinc, netinc, fcf, equity, assets are constant
    every quarter, every YoY growth feature is exactly zero (or -0)
    after warm-up."""
    result = compute_fundamentals_features(steady_panel).frame
    last = _last(result)
    growth_names = (
        "revenue_growth_yoy",
        "revenue_growth_qoq",
        "gross_profit_growth_yoy",
        "operating_income_growth_yoy",
        "net_income_growth_yoy",
        "fcf_growth_yoy",
        "equity_growth_yoy",
        "low_asset_growth_yoy",
    )
    for name in growth_names:
        assert last[name] == pytest.approx(0.0, abs=1e-12), name


def test_acceleration_is_delta_of_yoy() -> None:
    """Acceleration must be ``yoy[t] - yoy[t-4]`` exactly."""
    # Build a panel where revenue ramps: 100, 100, 100, 100, then 110...
    rows = []
    for idx in range(12):
        revenue = 100.0 if idx < 4 else 110.0 if idx < 8 else 121.0
        rows.append(_row("instrument-a", "AAA", idx, revenue=revenue))
    result = compute_fundamentals_features(_build_panel(rows)).frame.sort_values("datekey")

    # revenue_growth_yoy at idx=8 uses revenue_ttm[8]/revenue_ttm[4]-1 = (110*4)/(100*3+110)-1
    # Let pandas compute it directly so the test isn't an exercise in arithmetic.
    yoy = result["revenue_growth_yoy"].to_numpy()
    accel = result["revenue_growth_accel"].to_numpy()
    # accel must be NaN until both yoy[t] AND yoy[t-4] are defined.
    # yoy itself becomes defined at idx=4 (lag-4 of revenue_ttm available
    # from idx=4 onwards: revenue_ttm[4]/revenue_ttm[0]); accel needs an
    # additional 4 quarters so first valid at idx=8.
    assert all(np.isnan(accel[:8]))
    for i in range(8, 12):
        np.testing.assert_allclose(accel[i], yoy[i] - yoy[i - 4], rtol=1e-12)


def test_negative_premium_features_are_inverted(steady_panel: SharadarPanel) -> None:
    """The brief's positive-orientation contract: any feature whose raw
    economic interpretation is "lower is better" must be exported under
    a ``low_*`` name and sign-flipped."""
    result = compute_fundamentals_features(steady_panel).frame
    last = _last(result)
    # debt=100, equity=300, assets=500, cashneq=50, marketcap=1000.
    assert last["low_debt_to_equity"] == pytest.approx(-(100.0 / 300.0))
    assert last["low_debt_to_assets"] == pytest.approx(-(100.0 / 500.0))
    assert last["low_net_debt_to_marketcap"] == pytest.approx(-((100.0 - 50.0) / 1_000.0))


def test_low_share_issuance_is_negative_yoy_share_growth() -> None:
    rows = []
    for idx in range(5):
        sharesbas = 1_000_000 if idx < 4 else 1_100_000
        rows.append(_row("instrument-a", "AAA", idx, sharesbas=sharesbas))
    result = compute_fundamentals_features(_build_panel(rows)).frame
    last = _last(result)
    # YoY share-count growth at idx=4 is 1.1M / 1.0M - 1 = 0.1.
    assert last["low_share_issuance_yoy"] == pytest.approx(-0.1)


def test_low_accruals_legacy_formula(steady_panel: SharadarPanel) -> None:
    """Legacy formula carried over: -((netinc_ttm - ncfo_ttm) / assets_4q_avg)."""
    result = compute_fundamentals_features(steady_panel).frame
    last = _last(result)
    # netinc_ttm=40, ncfo_ttm=48, assets_4q_avg=500.
    expected = -((40.0 - 48.0) / 500.0)
    assert last["low_accruals_4q"] == pytest.approx(expected)


def test_dividend_yield_passthrough(steady_panel: SharadarPanel) -> None:
    result = compute_fundamentals_features(steady_panel).frame
    last = _last(result)
    assert last["dividend_yield"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# PIT / leakage
# ---------------------------------------------------------------------------


def test_ttm_features_are_nan_until_four_quarters() -> None:
    rows = [_row("instrument-a", "AAA", idx) for idx in range(4)]
    result = compute_fundamentals_features(_build_panel(rows)).frame.sort_values("datekey")
    # Only 4 rows → TTM aggregates become defined exactly at idx=3.
    assert result["roe_ttm"].iloc[:3].isna().all()
    assert pd.notna(result["roe_ttm"].iloc[3])


def test_yoy_features_are_nan_until_five_quarters() -> None:
    rows = [_row("instrument-a", "AAA", idx) for idx in range(5)]
    result = compute_fundamentals_features(_build_panel(rows)).frame.sort_values("datekey")
    # YoY on per-quarter equity requires equity at t and t-4 → first
    # valid index = 4.
    assert result["equity_growth_yoy"].iloc[:4].isna().all()
    assert pd.notna(result["equity_growth_yoy"].iloc[4])


def test_no_cross_instrument_leakage(two_instrument_panel: SharadarPanel) -> None:
    combined = compute_fundamentals_features(two_instrument_panel).frame
    bbb_only_rows = [
        row
        for row in two_instrument_panel.frame.to_dict(orient="records")
        if row["instrument_id"] == "instrument-b"
    ]
    bbb_panel = _build_panel(bbb_only_rows)
    bbb_only = compute_fundamentals_features(bbb_panel).frame

    combined_bbb = (
        combined.loc[combined["instrument_id"] == "instrument-b"]
        .sort_values("datekey")
        .reset_index(drop=True)
    )
    bbb_only = bbb_only.sort_values("datekey").reset_index(drop=True)

    for name in FEATURE_NAMES:
        pd.testing.assert_series_equal(
            combined_bbb[name],
            bbb_only[name],
            check_names=False,
        )


def test_coverage_counts_match_non_null_rows(steady_panel: SharadarPanel) -> None:
    result = compute_fundamentals_features(steady_panel)
    for name, coverage in result.coverage.items():
        assert coverage == int(result.frame[name].notna().sum()), name


# ---------------------------------------------------------------------------
# Sector neutralization
# ---------------------------------------------------------------------------


def test_sector_neutralization_via_post_processor(
    two_instrument_panel: SharadarPanel,
) -> None:
    """compute_fundamentals_features stays pure; sector neutralisation is
    a post-processor that takes a FeatureFrame and returns a FeatureFrame."""
    sector_map = {"instrument-a": "Tech", "instrument-b": "Tech"}
    raw = compute_fundamentals_features(two_instrument_panel)
    neutral = neutralize_feature_frame(raw, by="sector_median", sector_map=sector_map).frame
    # With two same-sector instruments, the per-date sector median is
    # the mean of the two values, and the residuals sum to zero.
    for _, group in neutral.groupby("datekey"):
        for name in FEATURE_NAMES:
            valid = group[name].dropna()
            if len(valid) == 2:
                # Two-element median = mean; residuals are symmetric.
                assert valid.sum() == pytest.approx(0.0, abs=1e-9), name


def test_neutralize_feature_frame_requires_sector_map(steady_panel: SharadarPanel) -> None:
    raw = compute_fundamentals_features(steady_panel)
    with pytest.raises(ValueError, match="sector_map"):
        neutralize_feature_frame(raw, by="sector_median")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_custom_version_round_trips() -> None:
    cfg = FundamentalsConfig(version="fundamentals-test-v0")
    assert cfg.version == "fundamentals-test-v0"


def test_custom_version_stamps_into_produced_specs(steady_panel: SharadarPanel) -> None:
    """A non-default config.version must flow into every FeatureSpec.version.

    Regression test for the silent-version-drop bug where
    compute_fundamentals_features used the module-level FEATURE_SPECS
    keyed to DEFAULT_CONFIG.version regardless of the config passed in.
    """
    cfg = FundamentalsConfig(version="fundamentals-test-v0")
    result = compute_fundamentals_features(steady_panel, config=cfg)
    for name in result.feature_names:
        assert result.feature_specs[name].version == "fundamentals-test-v0", name


def test_empty_version_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FundamentalsConfig(version="   ")
