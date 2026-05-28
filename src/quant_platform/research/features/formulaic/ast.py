"""Expression AST for the formulaic alpha factory.

Every alpha in the library is built as a programmatically-constructed
tree of :class:`Expression` nodes. The evaluator
(:func:`~.evaluator.evaluate_expression`) walks the tree to produce a
``pandas.Series``; introspection methods
(:meth:`Expression.required_inputs`, :meth:`Expression.lookback_days`,
:meth:`Expression.point_in_time`) walk the same tree to derive metadata
for the :class:`FeatureSpec` automatically — no hand-maintained sidecar
metadata.

Node types:

* :class:`Var` — a named input column from the
  :class:`~.panel.MarketPanel` (``close``, ``volume``, ``sector``, …).
* :class:`Const` — a scalar literal (int, float, or comparison-friendly
  string for ``group_rank`` group columns).
* :class:`UnaryOp` — ``-x``.
* :class:`BinOp` — ``+ - * /`` between two expressions or between
  expression and scalar.
* :class:`Compare` — ``< <= > >= == !=`` (used by :class:`Where`
  conditions).
* :class:`Where` — ternary ``where(cond, then_branch, else_branch)``,
  equivalent to NumPy's ``where``.
* :class:`OpCall` — a named operator from
  :mod:`~.operators` plus its argument list.

Every leaf carries enough information to compose introspection without
re-importing the operators module: ``Var`` knows the column name it
needs; ``OpCall`` knows its operator name plus per-arg lookback
contributions.

The :class:`Expression` base class overloads the Python arithmetic and
comparison operators so an alpha like

>>> Rank(Var("close")) * Rank(Var("volume"))

reads naturally. ``__hash__`` is provided so identical sub-expressions
share evaluator-cache entries (see :class:`.evaluator.ExpressionCache`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

ComparisonOp = Literal["<", "<=", ">", ">=", "==", "!="]
BinaryOp = Literal["+", "-", "*", "/"]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expression:
    """Base for every AST node.

    Subclasses override :meth:`required_inputs`, :meth:`lookback_days`,
    and :meth:`point_in_time`. The base class provides
    ``__add__``-and-friends so children of ``Expression`` compose with
    standard Python operators.
    """

    # ``ClassVar`` so dataclass __eq__ / __hash__ don't include it.
    _kind: ClassVar[str] = "expression"

    def required_inputs(self) -> frozenset[str]:
        """Names of panel columns this expression reads."""
        raise NotImplementedError

    def lookback_days(self) -> int:
        """Trading-day history this expression needs to produce a non-NaN value.

        Recurses through the AST; per-node contribution is added by the
        operator (e.g. ``delta(close, 5)`` contributes 5 to whatever its
        child needs).
        """
        raise NotImplementedError

    def point_in_time(self) -> bool:
        """Whether this expression is PIT-safe (no future-data leakage)."""
        raise NotImplementedError

    def walk(self) -> Iterable[Expression]:
        """Yield this node and every descendant. Default: just self."""
        yield self

    # ---- Python operator overloads ----

    def __neg__(self) -> UnaryOp:
        return UnaryOp("-", self)

    def __add__(self, other: Expression | float | int) -> BinOp:
        return BinOp("+", self, _coerce(other))

    def __radd__(self, other: Expression | float | int) -> BinOp:
        return BinOp("+", _coerce(other), self)

    def __sub__(self, other: Expression | float | int) -> BinOp:
        return BinOp("-", self, _coerce(other))

    def __rsub__(self, other: Expression | float | int) -> BinOp:
        return BinOp("-", _coerce(other), self)

    def __mul__(self, other: Expression | float | int) -> BinOp:
        return BinOp("*", self, _coerce(other))

    def __rmul__(self, other: Expression | float | int) -> BinOp:
        return BinOp("*", _coerce(other), self)

    def __truediv__(self, other: Expression | float | int) -> BinOp:
        return BinOp("/", self, _coerce(other))

    def __rtruediv__(self, other: Expression | float | int) -> BinOp:
        return BinOp("/", _coerce(other), self)

    # Comparison operators return :class:`Compare` nodes so they can
    # feed :class:`Where` conditions. Python's ``__eq__`` is reserved
    # by the dataclass for structural equality, so we expose ``eq()``
    # and ``ne()`` as explicit method calls (named ``cmp_eq`` /
    # ``cmp_ne`` to avoid shadowing) and overload only ``<``/``<=``/
    # ``>``/``>=`` which are not reserved.

    def __lt__(self, other: Expression | float | int) -> Compare:
        return Compare("<", self, _coerce(other))

    def __le__(self, other: Expression | float | int) -> Compare:
        return Compare("<=", self, _coerce(other))

    def __gt__(self, other: Expression | float | int) -> Compare:
        return Compare(">", self, _coerce(other))

    def __ge__(self, other: Expression | float | int) -> Compare:
        return Compare(">=", self, _coerce(other))

    def cmp_eq(self, other: Expression | float | int) -> Compare:
        return Compare("==", self, _coerce(other))

    def cmp_ne(self, other: Expression | float | int) -> Compare:
        return Compare("!=", self, _coerce(other))


def _coerce(value: Expression | float | int) -> Expression:
    """Lift Python scalars into :class:`Const` nodes so the AST is uniform."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float)):
        return Const(float(value))
    raise TypeError(
        f"Cannot coerce {type(value).__name__!r} into an AST node; "
        "expected Expression, int, or float."
    )


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Var(Expression):
    """A named input column read from the panel."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Var.name must be non-empty")

    def required_inputs(self) -> frozenset[str]:
        return frozenset({self.name})

    def lookback_days(self) -> int:
        return 0

    def point_in_time(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover — convenience
        return self.name


@dataclass(frozen=True)
class Const(Expression):
    """A scalar literal lifted into the AST."""

    value: float

    def required_inputs(self) -> frozenset[str]:
        return frozenset()

    def lookback_days(self) -> int:
        return 0

    def point_in_time(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover
        return repr(self.value)


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnaryOp(Expression):
    """Unary negation."""

    op: Literal["-"]
    operand: Expression

    def required_inputs(self) -> frozenset[str]:
        return self.operand.required_inputs()

    def lookback_days(self) -> int:
        return self.operand.lookback_days()

    def point_in_time(self) -> bool:
        return self.operand.point_in_time()

    def walk(self) -> Iterable[Expression]:
        yield self
        yield from self.operand.walk()


@dataclass(frozen=True)
class BinOp(Expression):
    """Binary arithmetic between two sub-expressions."""

    op: BinaryOp
    left: Expression
    right: Expression

    def required_inputs(self) -> frozenset[str]:
        return self.left.required_inputs() | self.right.required_inputs()

    def lookback_days(self) -> int:
        return max(self.left.lookback_days(), self.right.lookback_days())

    def point_in_time(self) -> bool:
        return self.left.point_in_time() and self.right.point_in_time()

    def walk(self) -> Iterable[Expression]:
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


@dataclass(frozen=True)
class Compare(Expression):
    """Element-wise comparison producing a boolean Series.

    Mostly used as the ``condition`` argument to :class:`Where`.
    """

    op: ComparisonOp
    left: Expression
    right: Expression

    def required_inputs(self) -> frozenset[str]:
        return self.left.required_inputs() | self.right.required_inputs()

    def lookback_days(self) -> int:
        return max(self.left.lookback_days(), self.right.lookback_days())

    def point_in_time(self) -> bool:
        return self.left.point_in_time() and self.right.point_in_time()

    def walk(self) -> Iterable[Expression]:
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


@dataclass(frozen=True)
class Where(Expression):
    """Element-wise ternary: ``where(cond, then_branch, else_branch)``."""

    condition: Expression
    then_branch: Expression
    else_branch: Expression

    def required_inputs(self) -> frozenset[str]:
        return (
            self.condition.required_inputs()
            | self.then_branch.required_inputs()
            | self.else_branch.required_inputs()
        )

    def lookback_days(self) -> int:
        return max(
            self.condition.lookback_days(),
            self.then_branch.lookback_days(),
            self.else_branch.lookback_days(),
        )

    def point_in_time(self) -> bool:
        return (
            self.condition.point_in_time()
            and self.then_branch.point_in_time()
            and self.else_branch.point_in_time()
        )

    def walk(self) -> Iterable[Expression]:
        yield self
        yield from self.condition.walk()
        yield from self.then_branch.walk()
        yield from self.else_branch.walk()


@dataclass(frozen=True)
class OpCall(Expression):
    """A named operator with positional arguments.

    Dispatch happens at :mod:`.evaluator` time via the
    :data:`~.operators.OPERATORS` registry. The AST node itself only
    carries the operator name, its argument list, and the operator's
    own ``window_lookback`` contribution (0 for non-windowed
    operators); the registry is the source of truth for *how* the
    operator computes.
    """

    name: str
    args: tuple[Expression, ...]
    window_lookback: int = 0
    int_args: tuple[int, ...] = field(default_factory=tuple)
    str_args: tuple[str, ...] = field(default_factory=tuple)
    float_args: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("OpCall.name must be non-empty")
        if self.window_lookback < 0:
            raise ValueError(
                f"OpCall({self.name!r}).window_lookback must be >= 0; got {self.window_lookback}"
            )

    def required_inputs(self) -> frozenset[str]:
        inputs: frozenset[str] = frozenset()
        for arg in self.args:
            inputs = inputs | arg.required_inputs()
        return inputs

    def lookback_days(self) -> int:
        child = max((arg.lookback_days() for arg in self.args), default=0)
        return child + self.window_lookback

    def point_in_time(self) -> bool:
        return all(arg.point_in_time() for arg in self.args)

    def walk(self) -> Iterable[Expression]:
        yield self
        for arg in self.args:
            yield from arg.walk()


__all__ = [
    "BinOp",
    "BinaryOp",
    "Compare",
    "ComparisonOp",
    "Const",
    "Expression",
    "OpCall",
    "UnaryOp",
    "Var",
    "Where",
]
