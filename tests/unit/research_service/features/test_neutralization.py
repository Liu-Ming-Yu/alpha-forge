"""Unit tests for cross-sectional neutralization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features import FeatureFrame, FeatureSpec
from quant_platform.research.features.neutralization import (
    attach_group_map,
    cross_sectional_rank,
    cross_sectional_zscore,
    neutralize_by_group,
    neutralize_by_size,
    neutralize_feature_frame,
)


def _two_date_panel() -> pd.DataFrame:
    """5-instrument, 2-date panel with deterministic feature values."""
    rows = []
    for date in (pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")):
        for idx, (inst, sector, mc, feat) in enumerate(
            [
                ("A", "Tech", 1e9, 1.0),
                ("B", "Tech", 5e9, 2.0),
                ("C", "Tech", 10e9, 3.0),
                ("D", "Energy", 2e9, 10.0),
                ("E", "Energy", 8e9, 20.0),
            ]
        ):
            del idx
            rows.append(
                {
                    "instrument_id": inst,
                    "date": date,
                    "sector": sector,
                    "marketcap": mc,
                    "feat": feat,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


def test_cross_sectional_rank_is_per_date() -> None:
    df = _two_date_panel()
    ranks = cross_sectional_rank(df, "feat", date_column="date", pct=True)
    df_with = df.assign(rank=ranks)
    # Per date, the largest feat ranks at 1.0 and the smallest at 0.2 = 1/5.
    for _, group in df_with.groupby("date"):
        assert group.loc[group["feat"].idxmax(), "rank"] == pytest.approx(1.0)
        assert group.loc[group["feat"].idxmin(), "rank"] == pytest.approx(0.2)


def test_cross_sectional_rank_preserves_nan() -> None:
    df = _two_date_panel()
    df.loc[df.index[0], "feat"] = np.nan
    ranks = cross_sectional_rank(df, "feat", date_column="date")
    assert pd.isna(ranks.iloc[0])
    # The remaining 4 observations on the first date now rank as
    # quartiles (0.25, 0.5, 0.75, 1.0); none are NaN.
    first_date_mask = df["date"] == df["date"].iloc[0]
    assert ranks[first_date_mask].dropna().count() == 4


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


def test_cross_sectional_zscore_is_mean_zero_per_date() -> None:
    df = _two_date_panel()
    z = cross_sectional_zscore(df, "feat", date_column="date")
    df_with = df.assign(z=z)
    for _, group in df_with.groupby("date"):
        # Population z-score (ddof=0) is exactly mean-zero per group.
        assert group["z"].mean() == pytest.approx(0.0, abs=1e-12)


def test_cross_sectional_zscore_winsorise_clips_extremes() -> None:
    # One absurdly large outlier should be clipped before standardising
    # so the remaining z-scores stay close to their winsorise-free values.
    df = _two_date_panel().iloc[:5].copy()
    df.loc[df.index[-1], "feat"] = 1e6  # outlier
    z_raw = cross_sectional_zscore(df, "feat", date_column="date")
    z_w = cross_sectional_zscore(df, "feat", date_column="date", winsorize=0.25)
    # Without winsorisation the outlier pulls every other z-score below
    # zero; with winsorisation, the remaining four are spread around zero
    # more symmetrically. Compare the std of the non-outlier rows.
    non_outlier = df.index[:-1]
    assert z_raw.loc[non_outlier].abs().mean() < 1.0
    assert z_w.loc[non_outlier].abs().mean() > z_raw.loc[non_outlier].abs().mean()


def test_cross_sectional_zscore_winsorise_validates_range() -> None:
    df = _two_date_panel()
    with pytest.raises(ValueError, match="winsorize"):
        cross_sectional_zscore(df, "feat", date_column="date", winsorize=0.6)


def test_cross_sectional_zscore_handles_constant_cross_section() -> None:
    df = _two_date_panel().iloc[:5].copy()
    df["feat"] = 7.0
    z = cross_sectional_zscore(df, "feat", date_column="date")
    # Std collapses to zero on this date — every z is NaN, not 0/0 = inf.
    assert z.isna().all()


# ---------------------------------------------------------------------------
# Group neutralization
# ---------------------------------------------------------------------------


def test_neutralize_by_group_subtracts_sector_median() -> None:
    df = _two_date_panel()
    out = neutralize_by_group(
        df,
        "feat",
        group_column="sector",
        date_column="date",
        statistic="median",
    )
    df_with = df.assign(resid=out)
    for (date, sector), group in df_with.groupby(["date", "sector"]):
        median = group["feat"].median()
        np.testing.assert_allclose(
            group["resid"].to_numpy(),
            (group["feat"] - median).to_numpy(),
            err_msg=f"sector {sector} on {date} did not centre to its median",
        )


def test_neutralize_by_group_mean_statistic() -> None:
    df = _two_date_panel()
    out = neutralize_by_group(
        df,
        "feat",
        group_column="sector",
        date_column="date",
        statistic="mean",
    )
    df_with = df.assign(resid=out)
    for _, group in df_with.groupby(["date", "sector"]):
        # Per-group mean of the residual is exactly zero.
        assert group["resid"].mean() == pytest.approx(0.0, abs=1e-12)


def test_neutralize_by_group_handles_missing_sector() -> None:
    df = _two_date_panel()
    df.loc[df.index[0], "sector"] = None
    # Must not raise; the null sector falls into a sentinel group rather
    # than being silently dropped.
    out = neutralize_by_group(
        df,
        "feat",
        group_column="sector",
        date_column="date",
    )
    assert out.notna().sum() > 0


def test_neutralize_by_group_rejects_bad_statistic() -> None:
    df = _two_date_panel()
    with pytest.raises(ValueError, match="neutralization statistic"):
        neutralize_by_group(
            df,
            "feat",
            group_column="sector",
            date_column="date",
            statistic="bogus",
        )


# ---------------------------------------------------------------------------
# Size neutralization
# ---------------------------------------------------------------------------


def test_neutralize_by_size_residual_is_orthogonal_to_log_size() -> None:
    df = _two_date_panel()
    residual = neutralize_by_size(df, "feat", date_column="date")
    df_with = df.assign(resid=residual, log_mc=np.log(df["marketcap"]))
    # Per-date OLS residual must be orthogonal to the regressor by
    # construction: corr(resid, log_size) ≈ 0.
    for date, group in df_with.groupby("date"):
        valid = group.dropna(subset=["resid", "log_mc"])
        if len(valid) < 2:
            continue
        corr = float(np.corrcoef(valid["resid"], valid["log_mc"])[0, 1])
        assert abs(corr) < 1e-10, f"date {date} residual not orthogonal: corr={corr}"


def test_neutralize_by_size_short_cross_section_returns_nan() -> None:
    df = _two_date_panel().iloc[:1].copy()
    residual = neutralize_by_size(df, "feat", date_column="date")
    # Single observation per date — OLS undefined, residual NaN.
    assert residual.isna().all()


def test_neutralize_by_size_constant_regressor_returns_centred() -> None:
    df = _two_date_panel().iloc[:5].copy()
    df["marketcap"] = 1e9  # constant size across the cross-section
    residual = neutralize_by_size(df, "feat", date_column="date")
    # OLS slope is undefined; helper falls back to mean-centring the
    # response.
    assert residual.mean() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# attach_group_map
# ---------------------------------------------------------------------------


def test_attach_group_map_fills_missing_with_sentinel() -> None:
    df = _two_date_panel().drop(columns=["sector"])
    mapping = {"A": "Tech", "B": "Tech", "D": "Energy"}  # C and E missing
    out = attach_group_map(df, group_map=mapping, output_column="sector")
    assert out["sector"].notna().all()
    assert (out.loc[out["instrument_id"] == "C", "sector"] == "__unknown__").all()
    assert (out.loc[out["instrument_id"] == "E", "sector"] == "__unknown__").all()


# ---------------------------------------------------------------------------
# neutralize_feature_frame
# ---------------------------------------------------------------------------


def _frame_spec(name: str, version: str = "test-v1") -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family="price_volume",
        description=f"{name} test",
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=1,
        version=version,
    )


def _feature_frame_from_panel(panel: pd.DataFrame, feature_name: str = "feat") -> FeatureFrame:
    return FeatureFrame(
        frame=panel.copy(),
        feature_names=(feature_name,),
        feature_specs={feature_name: _frame_spec(feature_name)},
        coverage={feature_name: int(panel[feature_name].notna().sum())},
        key_columns=("instrument_id", "date"),
    )


def test_neutralize_feature_frame_sector_median_zeros_pair_residual() -> None:
    panel = _two_date_panel().rename(columns={"feat": "feat"})
    frame = _feature_frame_from_panel(panel)
    sector_map = dict(zip(panel["instrument_id"], panel["sector"], strict=True))
    out = neutralize_feature_frame(frame, by="sector_median", sector_map=sector_map)
    # Same-sector pairs should sum to zero (median = mean for 2 elements,
    # residuals are symmetric).
    for (_date, _sector), group in out.frame.groupby(["date", "sector"]):
        valid = group["feat"].dropna()
        if len(valid) == 2:
            assert valid.sum() == pytest.approx(0.0, abs=1e-9)


def test_neutralize_feature_frame_size_residual_orthogonal_to_log_size() -> None:
    panel = _two_date_panel()
    frame = _feature_frame_from_panel(panel)
    out = neutralize_feature_frame(frame, by="size_residual", size_column="marketcap")
    # Per-date residual must be orthogonal to log(size) by OLS construction.
    out_df = out.frame.assign(log_mc=np.log(panel["marketcap"]))
    for _, group in out_df.groupby("date"):
        valid = group.dropna(subset=["feat", "log_mc"])
        if len(valid) >= 2:
            corr = float(np.corrcoef(valid["feat"], valid["log_mc"])[0, 1])
            assert abs(corr) < 1e-10


def test_neutralize_feature_frame_requires_sector_map_for_sector_kinds() -> None:
    panel = _two_date_panel()
    frame = _feature_frame_from_panel(panel)
    with pytest.raises(ValueError, match="sector_map"):
        neutralize_feature_frame(frame, by="sector_median")
    with pytest.raises(ValueError, match="sector_map"):
        neutralize_feature_frame(frame, by="sector_mean")


def test_neutralize_feature_frame_size_residual_validates_size_column() -> None:
    panel = _two_date_panel().drop(columns=["marketcap"])
    frame = _feature_frame_from_panel(panel)
    with pytest.raises(ValueError, match="size_column"):
        neutralize_feature_frame(frame, by="size_residual")


def test_neutralize_feature_frame_preserves_specs_and_keys() -> None:
    panel = _two_date_panel()
    frame = _feature_frame_from_panel(panel)
    sector_map = dict(zip(panel["instrument_id"], panel["sector"], strict=True))
    out = neutralize_feature_frame(frame, by="sector_median", sector_map=sector_map)
    assert out.feature_names == frame.feature_names
    assert out.feature_specs == frame.feature_specs
    assert out.key_columns == frame.key_columns
    # Coverage is recomputed against the residual but the keys match.
    assert set(out.coverage) == set(frame.coverage)
