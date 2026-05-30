"""Conviction-proportional weighting — the shared IC→Sharpe / transfer-coefficient kernel.

Sizing a long-only top-N book by *conviction* (how strongly the model ranks each
name) instead of equal weight raises the **transfer coefficient** in the
Fundamental Law of active management ``IR ≈ IC · √BR · TC`` — it transfers the
signal's conviction into the weights, converting a real-but-diffuse IC into a
higher information ratio (Sharpe) *without changing the selection or the IC*.

This function is the **single source of truth** for that math, deliberately in
``core`` so both sides use it and cannot diverge:

* the research backtest's ``ConvictionWeight`` weighting scheme
  (``services.research_service.campaigns.portfolio.weighting``) delegates here, and
* the live ``LongOnlyPortfolioConstructor`` (``core.algorithms.portfolio_construction``)
  calls it directly.

Parity-by-construction is not optional here: a live/backtest weighting mismatch
is exactly the failure mode that produced the dollar-volume scoring defect
(ADR-011). Keeping the arithmetic in one pure ``core`` helper makes that
mismatch impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Sweep-selected default shrinkage (peak Sharpe 1.0211 at 0.25; pure conviction
#: 0.0 over-concentrates → 0.956). See Arm Q in the latest-stack backtest.
DEFAULT_CONVICTION_SHRINKAGE = 0.25


def conviction_proportions(
    scores: Sequence[float],
    *,
    shrinkage: float = DEFAULT_CONVICTION_SHRINKAGE,
    reference: str = "min",
    risk: Sequence[float] | None = None,
) -> list[float]:
    """Return conviction-proportional weights (>= 0, summing to 1) for ``scores``.

    ``scores`` are the per-name model scores of the **already-selected** names
    (any order). The result is in the same order. The weights are::

        a_i  = max(0, score_i - ref)          # conviction above a reference
        a_i /= risk_i                          # optional risk-adjust (w ∝ α/d²)
        p_i  = a_i / Σ a                        # pure conviction proportions
        w_i  = shrinkage·(1/N) + (1-shrinkage)·p_i

    ``reference="min"`` subtracts the minimum selected score so the marginal
    name's conviction is ~0 and the strongest names carry the book;
    ``reference="zero"`` weights by the raw score. ``shrinkage`` ∈ [0, 1]
    interpolates toward equal weight (1.0 == equal weight) to blunt estimation
    error — the sweep shows some shrinkage is *required* (pure conviction
    over-concentrates). ``risk`` (optional) is a per-name positive divisor
    (e.g. idiosyncratic variance) for the book's ``w ∝ α/d²`` form; ``None``
    leaves the tilt unadjusted.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if reference not in ("min", "zero"):
        raise ValueError("reference must be 'min' or 'zero'")
    n = len(scores)
    if n == 0:
        return []
    ref = min(scores) if reference == "min" else 0.0
    conviction = [max(0.0, float(score) - ref) for score in scores]
    if risk is not None:
        if len(risk) != n:
            raise ValueError("risk must have the same length as scores")
        conviction = [c / r for c, r in zip(conviction, risk, strict=True)]
    total = sum(conviction)
    pure = [c / total for c in conviction] if total > 0 else [1.0 / n] * n
    equal = 1.0 / n
    return [shrinkage * equal + (1.0 - shrinkage) * p for p in pure]


__all__ = ["DEFAULT_CONVICTION_SHRINKAGE", "conviction_proportions"]
