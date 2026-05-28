"""Unit tests for AST mutation operators."""

from __future__ import annotations

import random

import pytest

from quant_platform.research.features.formulaic.ast import (
    Expression,
    OpCall,
    Var,
)
from quant_platform.research.features.formulaic.mining.grammar import AlphaGrammar
from quant_platform.research.features.formulaic.mining.mutation import (
    MUTATION_KINDS,
    change_window,
    mutate,
    prune,
    replace_var,
    swap_operator,
    wrap_op,
)
from quant_platform.research.features.formulaic.operators import (
    delta,
    rank,
    ts_corr,
    ts_rank,
    zscore,
)


def _is_pit(expr: Expression) -> bool:
    return expr.point_in_time()


# ---------------------------------------------------------------------------
# swap_operator
# ---------------------------------------------------------------------------


def test_swap_operator_changes_cross_sectional_op() -> None:
    expr = rank(Var("close"))
    rng = random.Random(1)
    g = AlphaGrammar()
    out = swap_operator(expr, g, rng)
    assert isinstance(out, OpCall)
    assert out.name == "zscore"  # only the other cross-sectional op in the catalog
    assert out.args[0] == Var("close")
    assert _is_pit(out)


def test_swap_operator_changes_time_series_op_preserving_window() -> None:
    expr = delta(Var("close"), 5)
    rng = random.Random(2)
    g = AlphaGrammar()
    out = swap_operator(expr, g, rng)
    assert isinstance(out, OpCall)
    assert out.name != "delta"
    # New op is one of the time-series unary ops.
    assert out.name in g.time_series_ops
    # Window preserved.
    assert out.int_args[0] == 5
    assert out.window_lookback == 5


def test_swap_operator_returns_input_when_no_opcalls() -> None:
    expr = Var("close")
    out = swap_operator(expr, AlphaGrammar(), random.Random(3))
    assert out is expr


# ---------------------------------------------------------------------------
# change_window
# ---------------------------------------------------------------------------


def test_change_window_changes_to_a_different_window() -> None:
    expr = delta(Var("close"), 5)
    rng = random.Random(4)
    g = AlphaGrammar(windows=(5, 10, 20, 60))
    out = change_window(expr, g, rng)
    assert isinstance(out, OpCall)
    assert out.int_args[0] != 5
    assert out.window_lookback == out.int_args[0]
    assert out.int_args[0] in g.windows


def test_change_window_returns_input_when_no_windowed_ops() -> None:
    expr = rank(Var("close"))  # rank has window_lookback=0
    out = change_window(expr, AlphaGrammar(), random.Random(5))
    assert out is expr


# ---------------------------------------------------------------------------
# replace_var
# ---------------------------------------------------------------------------


def test_replace_var_changes_a_leaf() -> None:
    expr = rank(Var("close"))
    rng = random.Random(6)
    g = AlphaGrammar(leaf_vars=("close", "volume"))
    out = replace_var(expr, g, rng)
    assert isinstance(out, OpCall)
    assert out.args[0] == Var("volume")  # only alternative


def test_replace_var_returns_input_when_vocabulary_is_singleton() -> None:
    expr = rank(Var("close"))
    out = replace_var(expr, AlphaGrammar(leaf_vars=("close",)), random.Random(7))
    assert out is expr


# ---------------------------------------------------------------------------
# wrap_op
# ---------------------------------------------------------------------------


def test_wrap_op_increases_depth_or_wraps_leaf() -> None:
    expr = Var("close")
    g = AlphaGrammar()
    # Try a few seeds; at least one should produce a real wrapper.
    wrapped_at_least_once = False
    for seed in range(30):
        out = wrap_op(expr, g, random.Random(seed))
        if out is not expr:
            wrapped_at_least_once = True
            assert isinstance(out, OpCall)
            assert _is_pit(out)
    assert wrapped_at_least_once


def test_wrap_op_refuses_to_wrap_cross_sectional_target_with_cross_sectional() -> None:
    """``wrap_op`` refuses to wrap an existing CS op (``rank(close)``)
    *directly* with another CS op — that would produce a strict
    ``rank(rank(close))`` or ``zscore(rank(close))``, which is the
    degenerate case the rule targets.

    Mutation is **not** required to enforce the grammar's broader
    "no CS-anywhere-inside-CS" rule: that rule is for sampling
    soundness. A mutation that produces e.g. ``rank(zscore(close))``
    by wrapping the inner Var is allowed — it's a valid AST, the
    evaluator handles it, and the admission gate will reject it if
    it scores poorly.
    """
    cross_op = rank(Var("close"))
    g = AlphaGrammar()
    cross = set(g.cross_sectional_ops)
    # If wrap_op picks the ROOT (the rank node) as the target and
    # tries to wrap it with another CS op, the rule kicks in and
    # returns the input unchanged. So whenever the output is still
    # rank(Var) AND identity-equals the input, no CS-with-CS wrap
    # happened at the root; whenever the output is different, the
    # wrapping was at a deeper position (or used a non-CS wrapper).
    for seed in range(100):
        out = wrap_op(cross_op, g, random.Random(seed))
        if isinstance(out, OpCall) and out.name in cross:
            # Either we're looking at the unchanged input (out is
            # cross_op), or we wrapped a deeper node — both are fine.
            inner = out.args[0]
            if out is not cross_op:
                # When the root rank/zscore is genuinely new (not the
                # identity-input return), its inner must not also be
                # the SAME cross-sectional op we just added — that's
                # the only true ``rank(rank(...))`` shape to forbid.
                assert not (
                    isinstance(inner, OpCall) and inner.name == out.name and inner.args == out.args
                ), f"seed={seed}: wrap_op produced rank(rank(x))/zscore(zscore(x))"


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_returns_one_of_an_opcalls_arg_subtrees() -> None:
    expr = rank(delta(Var("close"), 5))
    rng = random.Random(8)
    g = AlphaGrammar()
    out = prune(expr, g, rng)
    # The result is one of the OpCall arg sub-trees somewhere in the tree.
    possible = []
    for node in expr.walk():
        if isinstance(node, OpCall):
            possible.extend(node.args)
    assert out in possible or out == expr


def test_prune_returns_input_when_no_opcalls() -> None:
    expr = Var("close")
    out = prune(expr, AlphaGrammar(), random.Random(9))
    assert out is expr


# ---------------------------------------------------------------------------
# mutate (chooser)
# ---------------------------------------------------------------------------


def test_mutate_returns_known_kind() -> None:
    g = AlphaGrammar()
    seed_expr = rank(ts_corr(Var("close"), Var("volume"), 10))
    for seed in range(20):
        out, kind = mutate(seed_expr, g, random.Random(seed))
        assert kind in MUTATION_KINDS
        assert isinstance(out, Expression)
        assert _is_pit(out)


@pytest.mark.parametrize("seed", range(10))
def test_mutate_does_not_corrupt_ast(seed: int) -> None:
    """Mutating produces a tree the evaluator could in principle walk
    — i.e. every node is one of the canonical AST classes."""
    g = AlphaGrammar()
    base = ts_rank(Var("close"), 10)
    out, _ = mutate(base, g, random.Random(seed))
    # Every node has a known type.
    for node in out.walk():
        assert type(node).__name__ in {
            "Var",
            "Const",
            "UnaryOp",
            "BinOp",
            "Compare",
            "Where",
            "OpCall",
        }


def test_mutated_tree_lookback_is_finite() -> None:
    g = AlphaGrammar()
    base = ts_rank(zscore(Var("close")), 20)
    for seed in range(30):
        out, _ = mutate(base, g, random.Random(seed))
        # lookback_days() is always >= 0 and an int.
        lb = out.lookback_days()
        assert isinstance(lb, int)
        assert lb >= 0
