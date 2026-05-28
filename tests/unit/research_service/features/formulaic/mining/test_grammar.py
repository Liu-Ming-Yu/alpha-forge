"""Unit tests for :class:`AlphaGrammar`."""

from __future__ import annotations

import random

import pytest

from quant_platform.research.features.formulaic.ast import (
    BinOp,
    Const,
    Expression,
    OpCall,
    UnaryOp,
    Var,
)
from quant_platform.research.features.formulaic.mining.grammar import (
    DEFAULT_LEAF_VARS,
    DEFAULT_POWERS,
    DEFAULT_WINDOWS,
    AlphaGrammar,
)


def _is_valid_ast(expr: Expression) -> bool:
    """Smoke check: every leaf is Var or Const, no None children."""
    for node in expr.walk():
        if isinstance(node, Var) and not node.name.strip():
            return False
        if isinstance(node, (BinOp,)) and (node.left is None or node.right is None):
            return False  # pragma: no cover
        if isinstance(node, UnaryOp) and node.operand is None:
            return False  # pragma: no cover
        if isinstance(node, OpCall) and any(arg is None for arg in node.args):
            return False  # pragma: no cover
    return True


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_default_grammar_constructs_without_args() -> None:
    g = AlphaGrammar()
    assert g.leaf_vars == DEFAULT_LEAF_VARS
    assert g.windows == DEFAULT_WINDOWS
    assert g.powers == DEFAULT_POWERS
    assert g.max_depth >= 1
    assert g.max_total_lookback >= 0


def test_grammar_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        AlphaGrammar(max_depth=0)
    with pytest.raises(ValueError, match="max_total_lookback"):
        AlphaGrammar(max_total_lookback=-1)
    with pytest.raises(ValueError, match="leaf_vars"):
        AlphaGrammar(leaf_vars=())
    with pytest.raises(ValueError, match="windows"):
        AlphaGrammar(windows=())
    with pytest.raises(ValueError, match="powers"):
        AlphaGrammar(powers=())


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_produces_valid_ast() -> None:
    g = AlphaGrammar(max_depth=3, max_total_lookback=120)
    rng = random.Random(7)
    for _ in range(50):
        expr = g.sample(rng)
        assert _is_valid_ast(expr)
        # Every Var leaf must be from the grammar's vocabulary.
        for node in expr.walk():
            if isinstance(node, Var):
                assert node.name in g.leaf_vars


def test_sample_respects_max_total_lookback() -> None:
    """Every sample must fit under the configured lookback budget."""
    g = AlphaGrammar(max_depth=4, max_total_lookback=60, windows=(5, 10, 20))
    rng = random.Random(11)
    for _ in range(50):
        expr = g.sample(rng)
        assert expr.lookback_days() <= 60


def test_sample_is_reproducible_with_seeded_rng() -> None:
    """Same seed → same first N samples (each one drawn fresh from a
    deterministic RNG)."""
    g = AlphaGrammar(max_depth=3)
    a = [g.sample(random.Random(123)) for _ in range(5)]
    b = [g.sample(random.Random(123)) for _ in range(5)]
    assert a == b


def test_sample_uses_only_leaf_vars_from_vocabulary() -> None:
    g = AlphaGrammar(leaf_vars=("close", "volume"), max_depth=3)
    rng = random.Random(13)
    for _ in range(30):
        expr = g.sample(rng)
        for node in expr.walk():
            if isinstance(node, Var):
                assert node.name in ("close", "volume")


def test_sample_uses_only_window_sizes_from_vocabulary() -> None:
    g = AlphaGrammar(windows=(7, 14), max_depth=3)
    rng = random.Random(17)
    for _ in range(30):
        expr = g.sample(rng)
        for node in expr.walk():
            if isinstance(node, OpCall) and node.window_lookback > 0:
                assert node.window_lookback in (7, 14)


def test_sample_does_not_nest_cross_sectional_in_cross_sectional() -> None:
    """``rank(rank(x))`` or ``zscore(rank(x))`` would be degenerate."""
    g = AlphaGrammar(max_depth=4)
    rng = random.Random(19)
    cross = set(g.cross_sectional_ops)
    for _ in range(50):
        expr = g.sample(rng)

        # Walk the tree; flag any cross-sectional op whose ancestor
        # chain already contains a cross-sectional op.
        def _check_no_nested_cross(node: Expression, under_cs: bool) -> None:
            if isinstance(node, OpCall):
                if node.name in cross and under_cs:
                    raise AssertionError(f"nested cross-sectional: {node.name!r}")
                new_under = under_cs or node.name in cross
                for arg in node.args:
                    _check_no_nested_cross(arg, new_under)
                return
            if isinstance(node, BinOp):
                _check_no_nested_cross(node.left, under_cs)
                _check_no_nested_cross(node.right, under_cs)
                return
            if isinstance(node, UnaryOp):
                _check_no_nested_cross(node.operand, under_cs)

        _check_no_nested_cross(expr, under_cs=False)


def test_sample_exhaustion_raises_when_budget_too_tight() -> None:
    """A grammar whose smallest window already exceeds the budget
    raises after exhausting attempts."""
    g = AlphaGrammar(windows=(100,), max_total_lookback=10, max_depth=2)
    rng = random.Random(23)
    # Some seeds may yield a pure cross-sectional or element-wise tree
    # with no time-series windows (lookback=0). To force the rejection
    # branch we need a sample shape that requires a window; with depth=2
    # and the time-series weight dominant this is almost certain over
    # many seeds. The test is therefore "either every attempt fits the
    # budget, or the function raises" — both are correct behaviours.
    raised = False
    for _ in range(40):
        try:
            g.sample(rng, max_attempts=4)
        except RuntimeError:
            raised = True
            break
    # If never raised after 40 trials, fail loudly — the rejection
    # logic is unreachable, which would be a bug.
    assert raised, "Grammar with windows=(100,) should reject some samples"


def test_weighted_choice_sum_zero_raises() -> None:
    """Internal guard: zero-sum weights are a programmer error."""
    from quant_platform.research.features.formulaic.mining.grammar import _weighted_choice

    with pytest.raises(ValueError, match="sum to zero"):
        _weighted_choice(random.Random(0), [("a", 0.0), ("b", 0.0)])


def test_consts_appear_only_inside_signed_power() -> None:
    """The grammar shouldn't emit bare Const nodes as alpha roots; the
    only place a Const lives is inside ``signed_power(..., exponent)``
    where it's stored on ``float_args`` (not as an Expression child)."""
    g = AlphaGrammar(max_depth=3)
    rng = random.Random(29)
    for _ in range(30):
        expr = g.sample(rng)
        for node in expr.walk():
            assert not isinstance(node, Const), "Grammar should not sample bare Const leaves"
