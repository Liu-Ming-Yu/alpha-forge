"""Starter library of WorldQuant-style formulaic alphas.

Each library entry binds three things together:

* ``name`` — the exported feature column name (e.g. ``wq_alpha_001``).
* ``expression`` — the AST that computes it.
* ``description`` — a human-readable formula + intuition string that
  ends up on the generated :class:`FeatureSpec`.

The expressions are paraphrases of the WorldQuant 101 catalog's style
rather than verbatim transcriptions — the goal of this PR is to
exercise the engine end-to-end against a handful of representative
shapes, not to enumerate the full 101. Mining and the full catalog
land in follow-up sprints (brief Phase 4).

Every alpha is **evidence-gated** by default
(:data:`FeatureDirection.unknown` + ``larger_is_better=False``).
Downstream walk-forward / feature-audit code admits or retires
individual alphas based on out-of-sample IC; nothing in this module
declares an a-priori direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.research.features.formulaic.ast import Expression, Var
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
)

if TYPE_CHECKING:
    from quant_platform.research.features.contracts import FeatureDirection


@dataclass(frozen=True)
class FormulaicAlpha:
    """One library entry: name + expression + provenance."""

    name: str
    expression: Expression
    description: str
    expected_direction: FeatureDirection = "unknown"
    larger_is_better: bool = False


# ---------------------------------------------------------------------------
# Input shortcuts
# ---------------------------------------------------------------------------
#
# Var instances for the columns the starter library reads. Declared as
# module-level constants so the library file reads like a formula sheet
# rather than a parade of ``Var("close")`` calls.

_OPEN = Var("open")
_HIGH = Var("high")
_LOW = Var("low")
_CLOSE = Var("close")
_VOLUME = Var("volume")
_RETURNS = Var("returns")


# ---------------------------------------------------------------------------
# Library — 6 starter alphas
# ---------------------------------------------------------------------------

LIBRARY: tuple[FormulaicAlpha, ...] = (
    FormulaicAlpha(
        name="wq_alpha_001",
        expression=rank(ts_argmax(signed_power(absolute(_RETURNS), 2.0), 5)) - 0.5,
        description=(
            "rank(ts_argmax(signed_power(abs(returns), 2), 5)) - 0.5. "
            "WorldQuant #1 paraphrase: which day in the last 5 carried "
            "the largest squared-magnitude return? The inner abs strips "
            "the sign so a -5% day ranks as a 'big move' alongside a +5% "
            "day; without it, signed_power preserves sign and the alpha "
            "degenerates to 'position of most positive return'. The real "
            "WorldQuant #1 uses a ternary to inject magnitude via "
            "stddev — this paraphrase uses abs() for the same effect "
            "with one less operator."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_006",
        expression=-ts_corr(_OPEN, _VOLUME, 10),
        description=(
            "-ts_corr(open, volume, 10). Inverse 10-day rolling correlation "
            "between open price and volume; positive when price and volume "
            "move opposite each other, which is the divergence-reversal "
            "shape WorldQuant #6 captures."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_012",
        expression=sign(delta(_VOLUME, 1)) * -delta(_CLOSE, 1),
        description=(
            "sign(delta(volume, 1)) * -delta(close, 1). Reversal payoff "
            "scaled by the sign of yesterday's volume change — confirms a "
            "next-day mean-reversion bet only when volume moved in the "
            "expected direction. WorldQuant #12."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_026",
        expression=-ts_corr(rank(_VOLUME), rank(_HIGH), 5),
        description=(
            "-ts_corr(rank(volume), rank(high), 5). Rank-based decorrelation "
            "of volume and high; the rank step strips the scale and "
            "compares pure ordering. WorldQuant #26 paraphrase."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_041",
        expression=rank(_HIGH - _CLOSE) - rank(_HIGH - _OPEN),
        description=(
            "rank(high - close) - rank(high - open). Intraday tail "
            "asymmetry: ranks how far the close fell from the high vs "
            "how far the open opened below the high. WorldQuant #41 "
            "paraphrase — cross-sectional intraday-tail signal."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_101",
        expression=(_CLOSE - _OPEN) / (_HIGH - _LOW + 0.001),
        description=(
            "(close - open) / (high - low + 0.001). Intraday closing "
            "position within the day's range. The +0.001 prevents "
            "division-by-zero on halted symbols. WorldQuant #101 verbatim "
            "(modulo the epsilon)."
        ),
    ),
    FormulaicAlpha(
        name="wq_alpha_002_paraphrase",
        expression=-ts_corr(rank(delta(_VOLUME, 2)), rank((_CLOSE - _OPEN) / _OPEN), 6),
        description=(
            "-ts_corr(rank(delta(volume, 2)), rank((close - open) / open), 6). "
            "Volume-acceleration vs intraday-return rank decorrelation over "
            "6 days. WorldQuant #2 paraphrase."
        ),
    ),
    FormulaicAlpha(
        name="ts_zscore_close_20d",
        expression=ts_zscore(_CLOSE, 20),
        description=(
            "ts_zscore(close, 20). Per-instrument 20-day z-score of close. "
            "Larger absolute value = price further from its own 20d mean. "
            "Direction depends on regime (mean-reverting vs trending)."
        ),
    ),
    FormulaicAlpha(
        name="ts_rank_volume_60d",
        expression=ts_rank(absolute(decay_linear(_VOLUME, 5)), 60),
        description=(
            "ts_rank(abs(decay_linear(volume, 5)), 60). Where in the last "
            "60 days does today's decay-weighted recent volume sit? "
            "Captures volume spikes relative to a per-instrument recent "
            "baseline."
        ),
    ),
)


__all__ = [
    "LIBRARY",
    "FormulaicAlpha",
]
