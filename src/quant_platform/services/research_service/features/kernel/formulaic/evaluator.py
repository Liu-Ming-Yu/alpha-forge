"""Expression-tree evaluator for the formulaic alpha factory.

:func:`evaluate_expression` walks an :class:`~.ast.Expression` against
a :class:`~.panel.MarketPanel` and returns a ``pandas.Series`` aligned
to the panel's row index. The traversal is recursive; intermediate
results are memoised in an :class:`ExpressionCache` so common
sub-expressions (``rank(close)`` appearing twice in the same alpha)
evaluate once per call.

Dispatch
--------

* :class:`Var` → look up the named column in
  :attr:`MarketPanel.frame`. KeyError surfaces with the panel's
  available columns to help the contributor.
* :class:`Const` → broadcast the scalar value over the panel's index.
* :class:`UnaryOp` → ``-x``.
* :class:`BinOp` → ``x op y`` via pandas operators. Division goes
  through :func:`~..transforms.safe_div` so zero / negative / NaN
  denominators yield NaN rather than ``inf``.
* :class:`Compare` → ``x cmp y`` produces a boolean Series for
  :class:`Where` to consume.
* :class:`Where` → ``cond.where(then_branch, else_branch)``.
* :class:`OpCall` → :func:`.operators.dispatch` selects the compute
  function; the evaluator feeds it the evaluated arg Series plus the
  literal int/str/float arg tuples in declaration order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant_platform.services.research_service.features.kernel.formulaic.ast import (
    BinOp,
    Compare,
    Const,
    Expression,
    OpCall,
    UnaryOp,
    Var,
    Where,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import dispatch
from quant_platform.services.research_service.features.kernel.transforms import safe_div

if TYPE_CHECKING:
    from quant_platform.services.research_service.features.kernel.formulaic.panel import MarketPanel


class ExpressionCache:
    """Per-call memoisation of evaluated sub-expressions.

    Cache keys are :class:`Expression` instances themselves — frozen
    dataclasses with structural ``__eq__``/``__hash__``, so two
    AST-equivalent sub-expressions hash to the same key without our
    needing to canonicalise.
    """

    def __init__(self) -> None:
        self._store: dict[Expression, pd.Series] = {}

    def get(self, expr: Expression) -> pd.Series | None:
        return self._store.get(expr)

    def put(self, expr: Expression, value: pd.Series) -> None:
        self._store[expr] = value


def evaluate_expression(
    panel: MarketPanel,
    expr: Expression,
    *,
    cache: ExpressionCache | None = None,
) -> pd.Series:
    """Evaluate ``expr`` against ``panel`` and return a Series.

    Parameters
    ----------
    panel:
        Validated :class:`MarketPanel`.
    expr:
        AST node to evaluate.
    cache:
        Optional :class:`ExpressionCache`. Pass one in to share
        memoisation across multiple ``evaluate_expression`` calls in
        the same alpha-library compute pass.

    Returns
    -------
    pd.Series
        Float Series index-aligned to ``panel.frame``.
    """
    cache = cache if cache is not None else ExpressionCache()
    return _eval(panel, expr, cache)


def _eval(panel: MarketPanel, expr: Expression, cache: ExpressionCache) -> pd.Series:
    cached = cache.get(expr)
    if cached is not None:
        return cached
    result = _eval_uncached(panel, expr, cache)
    cache.put(expr, result)
    return result


def _eval_uncached(panel: MarketPanel, expr: Expression, cache: ExpressionCache) -> pd.Series:
    if isinstance(expr, Var):
        return _eval_var(panel, expr)
    if isinstance(expr, Const):
        return _eval_const(panel, expr)
    if isinstance(expr, UnaryOp):
        return -_eval(panel, expr.operand, cache)
    if isinstance(expr, BinOp):
        return _eval_binop(panel, expr, cache)
    if isinstance(expr, Compare):
        return _eval_compare(panel, expr, cache)
    if isinstance(expr, Where):
        return _eval_where(panel, expr, cache)
    if isinstance(expr, OpCall):
        return _eval_opcall(panel, expr, cache)
    raise TypeError(f"Unsupported expression node: {type(expr).__name__}")


def _eval_var(panel: MarketPanel, expr: Var) -> pd.Series:
    panel.require_column(expr.name)
    return panel.frame[expr.name].astype(float)


def _eval_const(panel: MarketPanel, expr: Const) -> pd.Series:
    return pd.Series(expr.value, index=panel.frame.index, dtype=float)


def _eval_binop(panel: MarketPanel, expr: BinOp, cache: ExpressionCache) -> pd.Series:
    left = _eval(panel, expr.left, cache)
    right = _eval(panel, expr.right, cache)
    if expr.op == "+":
        return (left + right).astype(float)
    if expr.op == "-":
        return (left - right).astype(float)
    if expr.op == "*":
        return (left * right).astype(float)
    if expr.op == "/":
        # ``require_positive_denom=False`` because alpha expressions
        # legitimately divide by quantities that can be negative
        # (``rank(open - close) / rank(high - low)`` for instance);
        # zero / NaN denominators still become NaN.
        return safe_div(left, right, require_positive_denom=False).astype(float)
    raise ValueError(f"Unsupported BinOp operator: {expr.op!r}")


def _eval_compare(panel: MarketPanel, expr: Compare, cache: ExpressionCache) -> pd.Series:
    left = _eval(panel, expr.left, cache)
    right = _eval(panel, expr.right, cache)
    op_map = {
        "<": pd.Series.lt,
        "<=": pd.Series.le,
        ">": pd.Series.gt,
        ">=": pd.Series.ge,
        "==": pd.Series.eq,
        "!=": pd.Series.ne,
    }
    func = op_map[expr.op]
    # pandas' boolean comparison ops treat NaN as False rather than
    # propagating it. That silently swaps NaN rows into the
    # ``else_branch`` when a downstream :class:`Where` consumes the
    # comparison; restore the "NaN in → NaN out" semantic so warm-up
    # rows in either operand stay NaN through the comparison.
    result = func(left, right).astype(float)
    valid = left.notna() & right.notna()
    return result.where(valid, np.nan)


def _eval_where(panel: MarketPanel, expr: Where, cache: ExpressionCache) -> pd.Series:
    cond = _eval(panel, expr.condition, cache)
    then_branch = _eval(panel, expr.then_branch, cache)
    else_branch = _eval(panel, expr.else_branch, cache)
    # ``cond`` is a float series with 1.0 (True), 0.0 (False), or NaN
    # (undefined). ``cond > 0`` returns False for NaN inputs (pandas
    # treats ``NaN > 0`` as False), which is what we want for the
    # then/else split; the final ``.where(cond.notna(), NaN)`` then
    # restores NaN on rows where the condition itself was undefined.
    out = then_branch.where(cond > 0, else_branch)
    return out.where(cond.notna(), np.nan).astype(float)


def _eval_opcall(panel: MarketPanel, expr: OpCall, cache: ExpressionCache) -> pd.Series:
    _, compute = dispatch(expr)
    evaluated_args: list[pd.Series] = [_eval(panel, arg, cache) for arg in expr.args]
    extra_args: tuple[Any, ...] = (*expr.int_args, *expr.str_args, *expr.float_args)
    return compute(panel, *evaluated_args, *extra_args).astype(float)


__all__ = [
    "ExpressionCache",
    "evaluate_expression",
]
