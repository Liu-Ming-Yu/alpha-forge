"""Drawdown experienced *during* the worst negative-IC streak.

This is the gate input for the drawdown-conditioned streak eligibility check
(ADR-004 Option D). The fixed ``fold_negative_ic_streak <= N`` gate is brittle —
its verdict hinges on a single fold and the threshold cannot be calibrated
out-of-sample (see ADR-004's held-out calibration section). The conditioned gate
instead asks: during the worst negative-IC streak, did the construction keep the
book inside the candidate drawdown bound? If so, a longer streak is tolerated
(up to a hard cap); if not, the strict floor applies. That requires knowing the
drawdown localized to the streak window, which this module computes.

The window is identified on the per-fold mean-IC series (path-independent), and
the drawdown is measured on the equity curve restricted to that window's folds —
fold-granular, consistent with the per-fold realized returns the evaluator
already produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    max_drawdown,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def worst_negative_streak_window(fold_ics: Sequence[float]) -> tuple[int, int] | None:
    """Return ``(start, end)`` inclusive indices of the longest run of
    consecutive strictly-negative values, or ``None`` if there is none.

    Ties resolve to the earliest run.
    """
    best: tuple[int, int] | None = None
    best_len = 0
    index = 0
    n = len(fold_ics)
    while index < n:
        if fold_ics[index] < 0:
            end = index
            while end < n and fold_ics[end] < 0:
                end += 1
            if end - index > best_len:
                best_len = end - index
                best = (index, end - 1)
            index = end
        else:
            index += 1
    return best


def max_drawdown_during_worst_streak(
    fold_ics: Sequence[float],
    fold_returns: Sequence[float],
) -> float:
    """Fold-granular drawdown over the worst negative-IC streak window.

    Returns ``0.0`` when there is no negative streak — there is no regime
    episode to gate, so the conditioned check imposes no extra drawdown demand.
    The drawdown is peak-to-trough on the equity curve built from the streak
    window's per-fold returns (``<= 0.0``), matching the sign convention of
    :func:`~...metrics.return_metrics.max_drawdown`.
    """
    window = worst_negative_streak_window(fold_ics)
    if window is None:
        return 0.0
    start, end = window
    return max_drawdown(list(fold_returns[start : end + 1]))


__all__ = ["max_drawdown_during_worst_streak", "worst_negative_streak_window"]
