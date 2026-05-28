"""AST mutation operators for evolutionary alpha search.

Five small, pure mutators that take an :class:`Expression` plus an
:class:`AlphaGrammar` and produce a new Expression with one
local change. Every mutator preserves the AST invariant (frozen
nodes, no cycles, every leaf is a Var or Const) and produces a
sample that the :mod:`..evaluator` can evaluate without raising.

* :func:`swap_operator` — replace an OpCall's compute with another
  of the same axis. ``rank(close)`` → ``zscore(close)``;
  ``delta(close, 5)`` → ``ts_zscore(close, 5)``.
* :func:`change_window` — replace the integer window of a windowed
  OpCall with another window from the grammar.
* :func:`replace_var` — swap a Var leaf for another from the grammar's
  ``leaf_vars`` vocabulary.
* :func:`wrap_op` — wrap an existing sub-tree in a new operator.
  ``close`` → ``rank(close)``.
* :func:`prune` — replace an OpCall with one of its arg sub-trees.
  ``rank(delta(close, 5))`` → ``delta(close, 5)``.

All mutators return the original expression when no valid mutation
applies (e.g. ``prune`` on a tree with no OpCall children). That
keeps the search loop free of None-checks.

Reproducibility
---------------

Every mutator takes an explicit ``rng: random.Random``. Combined with
the search algorithm's deterministic iteration order, the same seed
reproduces the full mutation chain across runs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.ast import (
    BinOp,
    Compare,
    Const,
    Expression,
    OpCall,
    UnaryOp,
    Var,
    Where,
)
from quant_platform.research.features.formulaic.operators import (
    absolute,
    decay_linear,
    delta,
    rank,
    sign,
    signed_power,
    ts_argmax,
    ts_rank,
    ts_zscore,
    zscore,
)

if TYPE_CHECKING:
    import random

    from quant_platform.research.features.formulaic.mining.grammar import (
        AlphaGrammar,
    )


# ---------------------------------------------------------------------------
# AST walking helpers
# ---------------------------------------------------------------------------


def _collect(root: Expression) -> list[Expression]:
    """Pre-order list of every node reachable from ``root``."""
    return list(root.walk())


def _replace_at(root: Expression, target: Expression, replacement: Expression) -> Expression:
    """Return a copy of ``root`` with ``target`` (identity-compared)
    replaced by ``replacement``.

    Walks the tree once. If ``target`` is not in the tree (by ``is``
    identity), returns ``root`` unchanged.
    """
    if root is target:
        return replacement
    if isinstance(root, Var):
        return root
    if isinstance(root, Const):
        return root
    if isinstance(root, UnaryOp):
        new_operand = _replace_at(root.operand, target, replacement)
        if new_operand is root.operand:
            return root
        return UnaryOp(root.op, new_operand)
    if isinstance(root, BinOp):
        new_left = _replace_at(root.left, target, replacement)
        new_right = _replace_at(root.right, target, replacement)
        if new_left is root.left and new_right is root.right:
            return root
        return BinOp(root.op, new_left, new_right)
    if isinstance(root, Compare):
        new_left = _replace_at(root.left, target, replacement)
        new_right = _replace_at(root.right, target, replacement)
        if new_left is root.left and new_right is root.right:
            return root
        return Compare(root.op, new_left, new_right)
    if isinstance(root, Where):
        new_c = _replace_at(root.condition, target, replacement)
        new_t = _replace_at(root.then_branch, target, replacement)
        new_e = _replace_at(root.else_branch, target, replacement)
        if new_c is root.condition and new_t is root.then_branch and new_e is root.else_branch:
            return root
        return Where(new_c, new_t, new_e)
    if isinstance(root, OpCall):
        new_args = tuple(_replace_at(arg, target, replacement) for arg in root.args)
        if all(new_a is old_a for new_a, old_a in zip(new_args, root.args, strict=True)):
            return root
        return replace(root, args=new_args)
    raise TypeError(f"Unsupported expression node in _replace_at: {type(root).__name__}")


# ---------------------------------------------------------------------------
# Per-mutator implementations
# ---------------------------------------------------------------------------


#: All mutation kinds — the string identifiers stored on
#: :class:`AutoAlphaProvenance.mutation_kind`. Listed here so contributors
#: adding a new mutator update one place.
MUTATION_KINDS: tuple[str, ...] = (
    "swap_operator",
    "change_window",
    "replace_var",
    "wrap_op",
    "prune",
)


def swap_operator(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    """Replace one OpCall with another of the same axis.

    Same-axis means cross-sectional → cross-sectional, time-series →
    time-series (preserving window when present), element-wise →
    element-wise. The structural shape (which args, which sub-trees)
    is preserved; only the operator identity changes.
    """
    op_calls = [node for node in _collect(expr) if isinstance(node, OpCall)]
    if not op_calls:
        return expr
    target = rng.choice(op_calls)

    cross = set(grammar.cross_sectional_ops)
    ts_unary = set(grammar.time_series_ops)
    ts_binary = set(grammar.time_series_binary_ops)
    elem = set(grammar.element_wise_ops)

    if target.name in cross:
        candidates = list(cross - {target.name})
        if not candidates:
            return expr
        new_name = rng.choice(candidates)
        replacement = _rebuild_unary_cross_sectional(new_name, target.args[0])
    elif target.name in ts_unary and len(target.args) == 1:
        candidates = list(ts_unary - {target.name})
        if not candidates:
            return expr
        new_name = rng.choice(candidates)
        window = target.int_args[0] if target.int_args else rng.choice(grammar.windows)
        replacement = _rebuild_unary_time_series(new_name, target.args[0], window)
    elif target.name in ts_binary and len(target.args) == 2:
        # ts_corr is the only binary ts op today; same-axis swap is a no-op.
        return expr
    elif target.name in elem:
        candidates = list(elem - {target.name})
        if not candidates:
            return expr
        new_name = rng.choice(candidates)
        replacement = _rebuild_element_wise(new_name, target.args[0], grammar, rng)
    else:
        return expr

    return _replace_at(expr, target, replacement)


def change_window(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    """Change the integer window of a windowed OpCall to another from
    the grammar's window catalog."""
    windowed = [
        node
        for node in _collect(expr)
        if isinstance(node, OpCall) and node.int_args and node.window_lookback > 0
    ]
    if not windowed:
        return expr
    target = rng.choice(windowed)
    candidates = [w for w in grammar.windows if w != target.int_args[0]]
    if not candidates:
        return expr
    new_window = rng.choice(candidates)
    replacement = replace(
        target,
        window_lookback=new_window,
        int_args=(new_window,) + target.int_args[1:],
    )
    return _replace_at(expr, target, replacement)


def replace_var(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    """Swap a Var leaf for another from the grammar's leaf vocabulary."""
    vars_in_tree = [node for node in _collect(expr) if isinstance(node, Var)]
    if not vars_in_tree:
        return expr
    target = rng.choice(vars_in_tree)
    candidates = [name for name in grammar.leaf_vars if name != target.name]
    if not candidates:
        return expr
    replacement = Var(rng.choice(candidates))
    return _replace_at(expr, target, replacement)


def wrap_op(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    """Wrap a random sub-tree in a new operator.

    Picks a random node (any kind) and wraps it. ``rank`` and
    ``zscore`` are only chosen when the target isn't already nested
    inside a cross-sectional operator (we don't generate
    ``rank(rank(...))``).
    """
    nodes = _collect(expr)
    if not nodes:
        return expr
    target = rng.choice(nodes)
    # ``Compare`` and ``Where`` aren't natural inner operands of the
    # numeric operators here. Skip them.
    if isinstance(target, (Compare, Where)):
        return expr

    # Choose a wrapper. Avoid stacking same-class wrappers blindly.
    wrapper = rng.choice(
        [
            "rank",
            "zscore",
            "ts_rank",
            "ts_zscore",
            "abs",
            "sign",
            "signed_power",
        ]
    )
    if wrapper in {"rank", "zscore"}:
        # Don't nest cross-sectional inside an existing cross-sectional.
        if isinstance(target, OpCall) and target.name in set(grammar.cross_sectional_ops):
            return expr
        replacement = rank(target) if wrapper == "rank" else zscore(target)
    elif wrapper == "ts_rank":
        replacement = ts_rank(target, rng.choice(grammar.windows))
    elif wrapper == "ts_zscore":
        replacement = ts_zscore(target, rng.choice(grammar.windows))
    elif wrapper == "abs":
        replacement = absolute(target)
    elif wrapper == "sign":
        replacement = sign(target)
    else:
        replacement = signed_power(target, rng.choice(grammar.powers))
    return _replace_at(expr, target, replacement)


def prune(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    """Replace an OpCall with one of its argument sub-trees.

    Lets the search shrink overly-deep ASTs that overfit the warm-up
    window. Operating on the root replaces the entire AST with one of
    its children.
    """
    del grammar  # not used; signature is uniform across mutators
    op_calls = [node for node in _collect(expr) if isinstance(node, OpCall) and node.args]
    if not op_calls:
        return expr
    target = rng.choice(op_calls)
    replacement = rng.choice(target.args)
    return _replace_at(expr, target, replacement)


# ---------------------------------------------------------------------------
# Rebuild helpers — keep the operator-builder API as the single source of
# truth for how each operator's OpCall is constructed (so window_lookback,
# str_args, float_args land in the right slots).
# ---------------------------------------------------------------------------


def _rebuild_unary_cross_sectional(name: str, child: Expression) -> Expression:
    if name == "rank":
        return rank(child)
    if name == "zscore":
        return zscore(child)
    raise ValueError(f"unknown cross-sectional op: {name!r}")


def _rebuild_unary_time_series(name: str, child: Expression, window: int) -> Expression:
    if name == "delta":
        return delta(child, window)
    if name == "ts_rank":
        return ts_rank(child, window)
    if name == "ts_zscore":
        return ts_zscore(child, window)
    if name == "decay_linear":
        return decay_linear(child, window)
    if name == "ts_argmax":
        return ts_argmax(child, window)
    raise ValueError(f"unknown time-series op: {name!r}")


def _rebuild_element_wise(
    name: str,
    child: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> Expression:
    if name == "abs":
        return absolute(child)
    if name == "sign":
        return sign(child)
    if name == "signed_power":
        return signed_power(child, rng.choice(grammar.powers))
    raise ValueError(f"unknown element-wise op: {name!r}")


# ---------------------------------------------------------------------------
# Random mutator chooser
# ---------------------------------------------------------------------------


_MUTATORS = {
    "swap_operator": swap_operator,
    "change_window": change_window,
    "replace_var": replace_var,
    "wrap_op": wrap_op,
    "prune": prune,
}


def mutate(
    expr: Expression,
    grammar: AlphaGrammar,
    rng: random.Random,
) -> tuple[Expression, str]:
    """Apply one randomly-chosen mutator. Returns ``(mutated, kind)``.

    ``kind`` is one of :data:`MUTATION_KINDS` and is the string the
    search loop records on :class:`AutoAlphaProvenance.mutation_kind`.

    Note: the mutator may return the input unchanged when no valid
    mutation applies (e.g. ``prune`` on a tree with no OpCall children
    other than the root). The caller is responsible for re-rolling if
    a fresh AST is required.
    """
    kind = rng.choice(MUTATION_KINDS)
    mutator = _MUTATORS[kind]
    return mutator(expr, grammar, rng), kind


__all__ = [
    "MUTATION_KINDS",
    "change_window",
    "mutate",
    "prune",
    "replace_var",
    "swap_operator",
    "wrap_op",
]
