"""Unit tests for :func:`evaluate_expression` and :class:`ExpressionCache`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import (
    Const,
    OpCall,
    Var,
    Where,
)
from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.formulaic.operators import (
    delta,
    rank,
)
from quant_platform.research.features.formulaic.panel import build_market_panel


def _bars(n_instruments: int = 3, n_rows: int = 30) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range(start="2024-01-02", periods=n_rows)
    for inst_idx in range(n_instruments):
        for i, d in enumerate(dates):
            close = 100.0 + 10 * inst_idx + i
            rows.append(
                {
                    "instrument_id": f"I{inst_idx}",
                    "date": d,
                    "open": close - 0.5,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_var_node_reads_from_panel() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(panel, Var("close"))
    np.testing.assert_array_equal(out.to_numpy(), panel.frame["close"].to_numpy())


def test_var_node_raises_on_missing_column() -> None:
    panel = build_market_panel(_bars())
    with pytest.raises(KeyError, match="not_a_real_column"):
        evaluate_expression(panel, Var("not_a_real_column"))


def test_const_broadcasts_over_index() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(panel, Const(7.5))
    assert (out == 7.5).all()
    assert len(out) == len(panel.frame)


def test_binop_arithmetic_matches_pandas() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(panel, Var("close") - Var("open"))
    np.testing.assert_allclose(
        out.to_numpy(),
        (panel.frame["close"] - panel.frame["open"]).to_numpy(),
    )


def test_binop_division_uses_safe_div() -> None:
    """Division by zero becomes NaN, not inf."""
    bars = _bars(n_instruments=1, n_rows=3)
    bars["high"] = [10.0, 10.0, 10.0]
    bars["low"] = [10.0, 10.0, 10.0]  # high - low == 0 every row
    panel = build_market_panel(bars)
    out = evaluate_expression(panel, Var("close") / (Var("high") - Var("low")))
    assert out.isna().all()
    assert not np.isinf(out.fillna(0)).any()


def test_unaryop_negates() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(panel, -Var("close"))
    np.testing.assert_allclose(out.to_numpy(), -panel.frame["close"].to_numpy())


def test_compare_produces_boolean_series_as_float() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(panel, Var("close") > Var("open"))
    # close > open always (close = open + 0.5 in the fixture).
    assert (out == 1.0).all()


def test_where_picks_branch_from_condition() -> None:
    panel = build_market_panel(_bars())
    out = evaluate_expression(
        panel,
        Where(
            condition=Var("close") > Const(110.0),
            then_branch=Var("close"),
            else_branch=Const(0.0),
        ),
    )
    df = panel.frame.assign(o=out)
    assert (df.loc[df["close"] > 110.0, "o"] == df.loc[df["close"] > 110.0, "close"]).all()
    assert (df.loc[df["close"] <= 110.0, "o"] == 0.0).all()


def test_expression_cache_avoids_recomputing_subtree() -> None:
    """When the same sub-expression appears twice in a parent, the
    cache hits and we don't recompute."""
    panel = build_market_panel(_bars())
    cache = ExpressionCache()
    common = rank(Var("close"))
    expr = common - common  # exact same node; cache should hit
    out = evaluate_expression(panel, expr, cache=cache)
    # Result is zero (a - a == 0) but also the cache must contain the
    # rank(close) node exactly once after the call.
    np.testing.assert_allclose(out.dropna().to_numpy(), 0.0)
    assert cache.get(common) is not None


def test_evaluator_rejects_unknown_opcall_name() -> None:
    panel = build_market_panel(_bars())
    bogus = OpCall(name="bogus_op", args=(Var("close"),))
    with pytest.raises(KeyError, match="unknown operator"):
        evaluate_expression(panel, bogus)


def test_evaluator_preserves_no_cross_instrument_leakage() -> None:
    """A time-series operator's first row should be NaN for each
    instrument, even when multiple instruments share dates."""
    panel = build_market_panel(_bars(n_instruments=3, n_rows=10))
    out = evaluate_expression(panel, delta(Var("close"), 2))
    df = panel.frame.assign(d=out)
    for _, g in df.groupby("instrument_id"):
        assert g["d"].iloc[:2].isna().all()
        assert g["d"].iloc[2:].notna().all()


# ---------------------------------------------------------------------------
# NaN propagation through Compare / Where
# ---------------------------------------------------------------------------


def test_compare_propagates_nan_through_either_operand() -> None:
    """A comparison where either side is NaN must return NaN, not False.

    Pandas' raw boolean ops treat NaN as False, which silently swapped
    warm-up rows into the else_branch of a Where. The compare evaluator
    now explicitly restores NaN-in → NaN-out semantics.
    """
    bars = _bars(n_instruments=1, n_rows=5)
    # Use a derived column where some rows are NaN: ``delta(close, 2)``
    # is NaN for the first two rows of the instrument.
    panel = build_market_panel(bars)
    expr = delta(Var("close"), 2) > Const(0.0)
    out = evaluate_expression(panel, expr)
    # First two rows must be NaN (because delta was NaN there).
    assert out.iloc[:2].isna().all()
    # Subsequent rows compare 2.0 > 0.0 = True → 1.0.
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), 1.0)


def test_where_propagates_nan_through_condition() -> None:
    """If the condition is NaN at a row, Where returns NaN — not the
    else_branch."""
    bars = _bars(n_instruments=1, n_rows=5)
    panel = build_market_panel(bars)
    expr = Where(
        # delta(close, 2) is NaN for the first 2 rows.
        condition=delta(Var("close"), 2) > Const(0.0),
        then_branch=Const(1.0),
        else_branch=Const(-1.0),
    )
    out = evaluate_expression(panel, expr)
    # First two rows must be NaN; after that, the condition is always
    # True (close is monotonically increasing), so the then_branch fires.
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), 1.0)


def test_where_picks_else_branch_when_condition_is_false() -> None:
    """Sanity: when cond is False (and non-NaN), we get the else_branch."""
    bars = _bars(n_instruments=1, n_rows=5)
    panel = build_market_panel(bars)
    # Compare close > 1e9 — always False, never NaN past the first row.
    expr = Where(
        condition=delta(Var("close"), 1) > Const(1e9),
        then_branch=Const(1.0),
        else_branch=Const(-1.0),
    )
    out = evaluate_expression(panel, expr)
    # Row 0: delta is NaN → cond NaN → out NaN.
    assert pd.isna(out.iloc[0])
    # Rows 1..4: cond is False → else_branch fires → -1.0.
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), -1.0)
