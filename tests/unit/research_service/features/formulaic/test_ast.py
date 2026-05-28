"""Unit tests for the formulaic AST.

Focus:
* Python operator overloads produce the right node types.
* ``required_inputs`` recurses correctly.
* ``lookback_days`` composes (parent windows ADD to child windows;
  binary ops take the MAX of their children).
* ``point_in_time`` propagates from leaves up.
* ``Const`` coercion catches type errors at construction.
"""

from __future__ import annotations

import pytest

from quant_platform.research.features.formulaic.ast import (
    BinOp,
    Compare,
    Const,
    OpCall,
    UnaryOp,
    Var,
)
from quant_platform.research.features.formulaic.operators import (
    delta,
    rank,
    ts_corr,
    ts_zscore,
)

# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


def test_var_metadata() -> None:
    v = Var("close")
    assert v.required_inputs() == frozenset({"close"})
    assert v.lookback_days() == 0
    assert v.point_in_time() is True


def test_var_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Var("   ")


def test_const_carries_no_inputs() -> None:
    c = Const(3.14)
    assert c.required_inputs() == frozenset()
    assert c.lookback_days() == 0


# ---------------------------------------------------------------------------
# Operator overloading
# ---------------------------------------------------------------------------


def test_binop_overloads_construct_correct_nodes() -> None:
    a, b = Var("close"), Var("open")
    assert isinstance(a + b, BinOp) and (a + b).op == "+"
    assert isinstance(a - b, BinOp) and (a - b).op == "-"
    assert isinstance(a * b, BinOp) and (a * b).op == "*"
    assert isinstance(a / b, BinOp) and (a / b).op == "/"


def test_unary_overload_produces_unaryop() -> None:
    a = Var("close")
    neg = -a
    assert isinstance(neg, UnaryOp)
    assert neg.op == "-"
    assert neg.operand == a


def test_compare_overloads_produce_compare_nodes() -> None:
    a = Var("close")
    assert isinstance(a < 0, Compare) and (a < 0).op == "<"
    assert isinstance(a > 0, Compare) and (a > 0).op == ">"
    assert isinstance(a <= 0, Compare) and (a <= 0).op == "<="
    assert isinstance(a >= 0, Compare) and (a >= 0).op == ">="
    # cmp_eq / cmp_ne — Python's __eq__ is reserved by the dataclass.
    assert isinstance(a.cmp_eq(0), Compare) and a.cmp_eq(0).op == "=="
    assert isinstance(a.cmp_ne(0), Compare) and a.cmp_ne(0).op == "!="


def test_scalar_coercion_lifts_into_const() -> None:
    a = Var("close")
    expr = a * 2
    assert isinstance(expr, BinOp)
    assert isinstance(expr.right, Const)
    assert expr.right.value == 2.0


def test_scalar_coercion_rejects_unsupported_types() -> None:
    a = Var("close")
    with pytest.raises(TypeError, match="Cannot coerce"):
        a + "not a number"  # type: ignore[operator]


def test_right_side_scalar_overloads() -> None:
    a = Var("close")
    expr = 5 + a
    assert isinstance(expr, BinOp)
    assert isinstance(expr.left, Const) and expr.left.value == 5.0
    assert expr.right == a


# ---------------------------------------------------------------------------
# Metadata recursion
# ---------------------------------------------------------------------------


def test_required_inputs_recurses_through_combinators() -> None:
    expr = Var("close") + Var("open") * Var("high")
    assert expr.required_inputs() == frozenset({"close", "open", "high"})


def test_lookback_days_takes_max_across_children() -> None:
    expr = delta(Var("close"), 5) - delta(Var("open"), 10)
    # delta(close, 5) needs 5; delta(open, 10) needs 10; binop takes max.
    assert expr.lookback_days() == 10


def test_lookback_days_adds_for_nested_opcalls() -> None:
    # ts_corr(window=6) of two delta(window=2) inputs needs 6 + 2 = 8.
    expr = ts_corr(delta(Var("volume"), 2), delta(Var("close"), 2), 6)
    assert expr.lookback_days() == 8


def test_lookback_does_not_add_for_cross_sectional_ops() -> None:
    # rank is window_lookback=0; only its child contributes.
    expr = rank(ts_zscore(Var("close"), 20))
    assert expr.lookback_days() == 20


def test_required_inputs_drops_constants() -> None:
    expr = Var("close") + 7
    assert expr.required_inputs() == frozenset({"close"})


def test_opcall_required_inputs_unions_args() -> None:
    expr = ts_corr(Var("high"), Var("volume"), 5)
    assert expr.required_inputs() == frozenset({"high", "volume"})


def test_point_in_time_propagates() -> None:
    # Every leaf in the starter library is PIT-true; composing PIT
    # children produces PIT parents.
    expr = rank(delta(Var("close"), 5) * Var("volume"))
    assert expr.point_in_time() is True


# ---------------------------------------------------------------------------
# Structural equality and hashability
# ---------------------------------------------------------------------------


def test_structurally_equal_expressions_hash_alike() -> None:
    a = rank(Var("close"))
    b = rank(Var("close"))
    assert a == b
    assert hash(a) == hash(b)


def test_opcall_rejects_negative_window_lookback() -> None:
    with pytest.raises(ValueError, match="window_lookback"):
        OpCall(name="bogus", args=(Var("close"),), window_lookback=-1)


def test_opcall_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OpCall(name="   ", args=(Var("close"),))
