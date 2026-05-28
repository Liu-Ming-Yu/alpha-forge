"""Template grammar for sampling random formulaic-alpha ASTs.

The miner's :mod:`.search.RandomSearch` and the seed population of
:mod:`.search.EvolutionarySearch` both need a way to produce a random
:class:`Expression` from scratch. This module defines that "way":

* :class:`AlphaGrammar` holds the operator catalog (split by axis),
  the variable vocabulary, the window-size candidates, and the depth
  budget.
* :meth:`AlphaGrammar.sample` draws one random AST from the grammar
  with a deterministic per-call RNG.

Type discipline
---------------

The grammar enforces a small handful of soundness rules so the
samples don't trivially evaluate to NaN or degenerate:

1. **No cross-sectional inside cross-sectional.** ``rank(rank(x))``
   is meaningless; the grammar refuses to nest cross-sectional
   operators inside each other.
2. **Time-series windows respect a lookback budget.** The caller's
   ``max_total_lookback`` cap is checked at the root so we don't
   generate alphas that need more history than the panel has.
3. **Leaf vocabulary is finite and curated.** Only ``Var`` names the
   caller registered as available + a small set of scalar literals
   for ``signed_power`` exponents. No string columns.
4. **Element-wise wrappers always wrap something.** We never sample
   bare ``Const(0.5)`` as the root of an alpha; the root is always a
   cross-sectional or time-series operator so the output varies
   across the panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.ast import (
    Expression,
    Var,
)
from quant_platform.research.features.formulaic.operators import (
    absolute,
    decay_linear,
    delta,
    rank,
    sign,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_rank,
    ts_zscore,
    zscore,
)

if TYPE_CHECKING:
    import random


#: Default Var names the grammar samples leaf positions from. Match
#: the columns :func:`~..panel.build_market_panel` always provides
#: (raw OHLCV + derived returns / dollar_volume).
DEFAULT_LEAF_VARS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "dollar_volume",
)

#: Default rolling-window candidates. Chosen to span the brief's
#: time horizons (~1 week, ~1 month, ~3 months) and to compose
#: cleanly with the existing operator builders.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

#: Default ``signed_power`` exponents. Limited to a small palette so
#: the grammar doesn't waste budget on near-identical tail-shapes.
DEFAULT_POWERS: tuple[float, ...] = (0.5, 2.0, 3.0)


@dataclass(frozen=True)
class AlphaGrammar:
    """Configuration for random alpha sampling.

    Attributes
    ----------
    leaf_vars:
        Var names allowed at leaf positions.
    windows:
        Window-size candidates for ``ts_*`` / ``delta`` / ``delay`` /
        ``decay_linear`` / ``ts_argmax``.
    powers:
        Exponent candidates for ``signed_power``.
    max_depth:
        Hard cap on AST depth. Stops runaway recursion.
    max_total_lookback:
        Reject samples whose total ``lookback_days`` exceeds this
        cap. Lets the caller align the grammar with the panel's
        available history.
    """

    leaf_vars: tuple[str, ...] = DEFAULT_LEAF_VARS
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    powers: tuple[float, ...] = DEFAULT_POWERS
    max_depth: int = 4
    max_total_lookback: int = 252
    # Sampling weights are stored as separate ratios so callers can
    # nudge the search toward a particular shape (e.g. heavier on
    # cross-sectional ranks). The numerator only matters relative to
    # the sum across the four kinds at the same depth.
    weight_cross_sectional: float = 1.0
    weight_time_series: float = 1.5
    weight_binop: float = 1.0
    weight_element_wise: float = 0.5

    # Per-class lists; declared as constants here so a contributor
    # adding a new operator updates one place.
    cross_sectional_ops: tuple[str, ...] = ("rank", "zscore")
    time_series_ops: tuple[str, ...] = (
        "delta",
        "ts_rank",
        "ts_zscore",
        "decay_linear",
        "ts_argmax",
    )
    time_series_binary_ops: tuple[str, ...] = ("ts_corr",)
    element_wise_ops: tuple[str, ...] = ("abs", "sign", "signed_power")
    binary_ops: tuple[str, ...] = ("+", "-", "*", "/")

    # Internal flags carrying the "is the current subtree wrapped in a
    # cross-sectional op already?" state — toggled inside _sample.
    # Default ``False`` is the public entry point's starting state.
    _under_cross_sectional: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("AlphaGrammar.max_depth must be >= 1")
        if self.max_total_lookback < 0:
            raise ValueError("AlphaGrammar.max_total_lookback must be >= 0")
        if not self.leaf_vars:
            raise ValueError("AlphaGrammar.leaf_vars must be non-empty")
        if not self.windows:
            raise ValueError("AlphaGrammar.windows must be non-empty")
        if not self.powers:
            raise ValueError("AlphaGrammar.powers must be non-empty")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: random.Random, *, max_attempts: int = 32) -> Expression:
        """Return one random AST.

        Retries up to ``max_attempts`` times if a sample exceeds the
        lookback budget. Raises :class:`RuntimeError` if every attempt
        fails — usually means the budget is too tight for the
        configured windows.
        """
        for _ in range(max_attempts):
            expr = self._sample_op(rng, depth=0, under_cross_sectional=False)
            if expr.lookback_days() <= self.max_total_lookback:
                return expr
        raise RuntimeError(
            f"AlphaGrammar.sample exhausted {max_attempts} attempts without "
            f"finding a sample under max_total_lookback={self.max_total_lookback}. "
            "Increase max_total_lookback or shrink the windows tuple."
        )

    # The actual recursion. Returns an Expression at depth ``depth``.
    # ``under_cross_sectional`` is the type-discipline switch from the
    # module docstring rule #1.
    def _sample_op(
        self,
        rng: random.Random,
        *,
        depth: int,
        under_cross_sectional: bool,
    ) -> Expression:
        if depth >= self.max_depth:
            return self._sample_leaf(rng)

        # Pick a "kind" weighted by the grammar config. Cross-sectional
        # operators drop out when we're already inside one.
        kinds: list[tuple[str, float]] = [
            ("time_series", self.weight_time_series),
            ("binop", self.weight_binop),
            ("element_wise", self.weight_element_wise),
        ]
        if not under_cross_sectional:
            kinds.append(("cross_sectional", self.weight_cross_sectional))

        kind = _weighted_choice(rng, kinds)
        if kind == "cross_sectional":
            return self._sample_cross_sectional(rng, depth=depth)
        if kind == "time_series":
            return self._sample_time_series(
                rng, depth=depth, under_cross_sectional=under_cross_sectional
            )
        if kind == "binop":
            return self._sample_binop(rng, depth=depth, under_cross_sectional=under_cross_sectional)
        return self._sample_element_wise(
            rng, depth=depth, under_cross_sectional=under_cross_sectional
        )

    def _sample_leaf(self, rng: random.Random) -> Var:
        return Var(rng.choice(self.leaf_vars))

    def _sample_cross_sectional(self, rng: random.Random, *, depth: int) -> Expression:
        op = rng.choice(self.cross_sectional_ops)
        child = self._sample_op(rng, depth=depth + 1, under_cross_sectional=True)
        if op == "rank":
            return rank(child)
        return zscore(child)

    def _sample_time_series(
        self,
        rng: random.Random,
        *,
        depth: int,
        under_cross_sectional: bool,
    ) -> Expression:
        # 30% chance of picking the binary ts_corr; otherwise one of
        # the unary time-series operators.
        if rng.random() < 0.3 and self.time_series_binary_ops:
            op = rng.choice(self.time_series_binary_ops)
            window = rng.choice(self.windows)
            left = self._sample_op(
                rng, depth=depth + 1, under_cross_sectional=under_cross_sectional
            )
            right = self._sample_op(
                rng, depth=depth + 1, under_cross_sectional=under_cross_sectional
            )
            if op == "ts_corr":
                return ts_corr(left, right, window)
        op = rng.choice(self.time_series_ops)
        window = rng.choice(self.windows)
        child = self._sample_op(rng, depth=depth + 1, under_cross_sectional=under_cross_sectional)
        if op == "delta":
            return delta(child, window)
        if op == "ts_rank":
            return ts_rank(child, window)
        if op == "ts_zscore":
            return ts_zscore(child, window)
        if op == "decay_linear":
            return decay_linear(child, window)
        return ts_argmax(child, window)

    def _sample_binop(
        self,
        rng: random.Random,
        *,
        depth: int,
        under_cross_sectional: bool,
    ) -> Expression:
        op = rng.choice(self.binary_ops)
        left = self._sample_op(rng, depth=depth + 1, under_cross_sectional=under_cross_sectional)
        right = self._sample_op(rng, depth=depth + 1, under_cross_sectional=under_cross_sectional)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        return left / right

    def _sample_element_wise(
        self,
        rng: random.Random,
        *,
        depth: int,
        under_cross_sectional: bool,
    ) -> Expression:
        op = rng.choice(self.element_wise_ops)
        child = self._sample_op(rng, depth=depth + 1, under_cross_sectional=under_cross_sectional)
        if op == "abs":
            return absolute(child)
        if op == "sign":
            return sign(child)
        # signed_power
        exponent = rng.choice(self.powers)
        return signed_power(child, exponent)


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    """Pick one option key with weights proportional to its float value."""
    total = sum(weight for _, weight in options)
    if total <= 0.0:
        raise ValueError("AlphaGrammar: weighted-choice options sum to zero")
    pick = rng.random() * total
    cumulative = 0.0
    for name, weight in options:
        cumulative += weight
        if pick <= cumulative:
            return name
    return options[-1][0]


__all__ = [
    "DEFAULT_LEAF_VARS",
    "DEFAULT_POWERS",
    "DEFAULT_WINDOWS",
    "AlphaGrammar",
]
