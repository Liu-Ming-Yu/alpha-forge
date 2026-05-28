"""Unit tests for the formulaic operator library.

Each test pins one operator's correctness on a small synthetic panel
where the expected output is easy to hand-derive. The fixtures avoid
floating-point chaos by using monotonic price/volume sequences so
ranks and z-scores collapse to closed-form expectations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.evaluator import evaluate_expression
from quant_platform.research.features.formulaic.operators import (
    absolute,
    decay_linear,
    delay,
    delta,
    group_rank,
    rank,
    sign,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_rank,
    ts_zscore,
    zscore,
)
from quant_platform.research.features.formulaic.panel import build_market_panel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bars(
    instruments: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    n_rows: int = 100,
    start: str = "2023-01-02",
    *,
    sector: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a multi-instrument bar frame whose closes are deterministic."""
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range(start=start, periods=n_rows)
    for inst_idx, inst in enumerate(instruments):
        base = 100.0 + 50.0 * inst_idx
        for i, d in enumerate(dates):
            close = base + i
            rows.append(
                {
                    "instrument_id": inst,
                    "date": d,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000 + 10 * i + 100 * inst_idx,
                }
            )
    df = pd.DataFrame(rows)
    if sector is not None:
        df["sector"] = df["instrument_id"].map(sector)
    return df


@pytest.fixture
def panel() -> object:
    return build_market_panel(_bars())


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------


def test_delta_is_exact_difference(panel: object) -> None:
    out = evaluate_expression(panel, delta(Var("close"), 5))
    # Within an instrument, close grows linearly by +1/day, so delta=5 always.
    valid = out.dropna()
    assert (valid == 5.0).all()


def test_delay_shifts_per_instrument(panel: object) -> None:
    out = evaluate_expression(panel, delay(Var("close"), 1))
    # In each instrument the first row has no lag → NaN; everything else
    # equals the prior row's close.
    df = panel.frame
    for _, g in df.groupby("instrument_id"):
        idx = g.index
        # First row NaN.
        assert pd.isna(out.iloc[idx[0]])
        # Subsequent rows equal the previous row's close.
        np.testing.assert_array_equal(
            out.loc[idx[1:]].to_numpy(),
            g["close"].iloc[:-1].to_numpy(),
        )


def test_ts_rank_is_percentile_in_window(panel: object) -> None:
    out = evaluate_expression(panel, ts_rank(Var("close"), 10))
    # In each instrument, close is monotonically increasing, so today is
    # always the max in any trailing window → rank-pct == 1.0.
    valid = out.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 1.0)


def test_ts_zscore_normalises_per_instrument(panel: object) -> None:
    out = evaluate_expression(panel, ts_zscore(Var("close"), 20))
    # For a +1/day arithmetic sequence of length 20, the last point is
    # 9.5 above the rolling mean, and the sample std (ddof=1) is
    # sqrt(sum((x - mean)^2) / 19) = sqrt(665 / 19) = sqrt(35) ≈ 5.9161.
    # So the z-score at the last row is 9.5 / sqrt(35) ≈ 1.60579.
    valid = out.dropna()
    expected_z = 9.5 / (35.0**0.5)
    assert valid.iloc[-1] == pytest.approx(expected_z, rel=1e-9)


def test_ts_corr_is_exactly_one_for_identical_series(panel: object) -> None:
    out = evaluate_expression(panel, ts_corr(Var("close"), Var("close"), 10))
    # corr(x, x) = 1 wherever the rolling std is defined.
    valid = out.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 1.0, atol=1e-10)


def test_decay_linear_collapses_to_constant_when_input_is_constant() -> None:
    bars = _bars(instruments=("AAA",), n_rows=30)
    bars["close"] = 50.0  # constant
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, decay_linear(Var("close"), 5))
    valid = out.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 50.0, atol=1e-12)


def test_decay_linear_weights_sum_to_one() -> None:
    # If decay-weighted average of (0, 0, …, 0, x) is x * w_recent, then
    # the most recent weight equals the operator's "recency emphasis"
    # number. With weights = (1/15, 2/15, …, 5/15) for window=5 the
    # most recent gets 5/15 = 1/3.
    bars = _bars(instruments=("AAA",), n_rows=10)
    bars["close"] = [0.0] * 9 + [15.0]
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, decay_linear(Var("close"), 5))
    # The decay_linear value at the last row uses weights normalised to
    # sum to 1; for input zeros except the most-recent (= 15), the
    # weighted average equals 15 * 5/15 = 5.
    assert out.iloc[-1] == pytest.approx(5.0)


def test_ts_argmax_returns_position_in_one_indexed_window(panel: object) -> None:
    out = evaluate_expression(panel, ts_argmax(Var("close"), 5))
    # close is monotonically increasing within an instrument, so the
    # max in any 5-row window is always the most-recent row → position 1.
    valid = out.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------


def test_rank_is_per_date_pct(panel: object) -> None:
    out = evaluate_expression(panel, rank(Var("close")))
    # Across 4 instruments on any date, ranks should be {0.25, 0.5,
    # 0.75, 1.0}. Verify the set on the last date.
    df = panel.frame.assign(rank_value=out)
    last_date = df["date"].max()
    last_ranks = df.loc[df["date"] == last_date, "rank_value"].sort_values().to_numpy()
    np.testing.assert_allclose(last_ranks, [0.25, 0.5, 0.75, 1.0])


def test_zscore_is_mean_zero_per_date(panel: object) -> None:
    out = evaluate_expression(panel, zscore(Var("close")))
    df = panel.frame.assign(z=out)
    for _, g in df.groupby("date"):
        if g["z"].notna().sum() >= 2:
            # Population std (ddof=0) zscore is mean-zero per date.
            assert g["z"].mean() == pytest.approx(0.0, abs=1e-9)


def test_group_rank_is_per_date_per_group() -> None:
    sector = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy"}
    bars = _bars(instruments=tuple(sector.keys()), n_rows=10, sector=sector)
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, group_rank(Var("close"), "sector"))
    # Two instruments per sector → ranks are 0.5 and 1.0 within each
    # sector on every date.
    df = panel.frame.assign(r=out)
    for (_date, _sec), g in df.groupby(["date", "sector"]):
        ranks = sorted(g["r"].to_numpy())
        np.testing.assert_allclose(ranks, [0.5, 1.0])


def test_group_rank_requires_group_column() -> None:
    bars = _bars(instruments=("AAA",), n_rows=3)
    panel = build_market_panel(bars)
    with pytest.raises(KeyError, match="sector"):
        evaluate_expression(panel, group_rank(Var("close"), "sector"))


# ---------------------------------------------------------------------------
# Element-wise operators
# ---------------------------------------------------------------------------


def test_absolute_preserves_magnitude() -> None:
    bars = _bars(instruments=("AAA",), n_rows=5)
    bars["close"] = [-3.0, -1.0, 0.0, 2.0, 4.0]
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, absolute(Var("close")))
    np.testing.assert_allclose(out.to_numpy(), [3.0, 1.0, 0.0, 2.0, 4.0])


def test_sign_returns_minus_zero_plus() -> None:
    bars = _bars(instruments=("AAA",), n_rows=5)
    bars["close"] = [-2.0, -0.0, 0.0, 1.5, 7.0]
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, sign(Var("close")))
    np.testing.assert_allclose(out.to_numpy(), [-1.0, 0.0, 0.0, 1.0, 1.0])


def test_signed_power_preserves_sign_compresses_magnitude() -> None:
    bars = _bars(instruments=("AAA",), n_rows=5)
    bars["close"] = [-4.0, -1.0, 0.0, 1.0, 9.0]
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, signed_power(Var("close"), 0.5))
    expected = [-2.0, -1.0, 0.0, 1.0, 3.0]  # sign * sqrt(|x|)
    np.testing.assert_allclose(out.to_numpy(), expected)


# ---------------------------------------------------------------------------
# No cross-instrument leakage
# ---------------------------------------------------------------------------


def test_delta_does_not_leak_across_instruments() -> None:
    # First row of each instrument's delta MUST be NaN — if it weren't,
    # the rolling shift had crossed an instrument boundary.
    bars = _bars(instruments=("AAA", "BBB"), n_rows=10)
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, delta(Var("close"), 3))
    df = panel.frame.assign(d=out)
    for _, g in df.groupby("instrument_id"):
        # First 3 rows are NaN; the rest are non-NaN.
        assert g["d"].iloc[:3].isna().all()
        assert g["d"].iloc[3:].notna().all()
