"""Fold-streak-aware exposure throttle.

When walk-forward evidence shows recent out-of-sample folds with
negative mean IC, scale down the production-candidate exposure for
the next fold. Composable with the existing ``FoldVolatilityScale``
— both compute a multiplicative scale on the same exposure axis;
the effective scale is their product.

**Why this exists.** ADR-003's corrected backtest left the latest
stack failing eligibility on ``fold_negative_ic_streak`` (4–7 vs
the ≤2 gate). The IC signal isn't dead (bootstrap CIs are positive
for the formulaic arm), but there are 4–7-month stretches where
the rank correlation flips — classic regime drift. This module
attacks that constraint directly using already-evidenced OOS
signal: a fold-level EWMA smooth throttle plus a hard
consecutive-negative-streak circuit breaker.

**PIT contract.** The scale at fold ``N`` is computed from the
mean OOS ICs of folds ``0..N-1`` only. The caller appends fold
``N``'s realized IC to the history **after** evaluating fold ``N``,
never before — otherwise the scale would leak the current fold's
OOS information into its own exposure decision.

**Pure throttle, not a dial.** The scale lives in ``[0, 1]``: we
can only cut exposure, never boost. The goal here is to pass an
eligibility gate, not to maximize. An asymmetric (>1.0) version
would be an offensive variant for a later PR.

**What this module CANNOT do.** It is a *reactive* regime-risk
overlay: exposure drops only *after* a streak of bad OOS folds has
already been observed. It will:

* Reduce damage during persistent adverse regimes (the v3
  universe-300 run bench'd 18 of 63 folds, dropping turnover by
  43% and giving Arm E a +8.8% Sharpe vs Arm D).
* **Lag sharp regime recoveries** — when IC flips back to
  positive after a bad streak, the EWMA takes a few folds to
  catch up and the dial keeps exposure suppressed during the
  early-recovery period.
* **Cannot improve IC.** The dial throttles exposure, not
  predictions; ``fold_negative_ic_streak`` is invariant to
  exposure decisions. Moving that gate requires changing the
  alpha itself (regime overlay, new feature family, etc.).

Reach for the dial when you want a defensible defensive layer on a
production-candidate arm; reach for a regime overlay (ADR-005-class
work) when you want predictions that *adapt* rather than exposure
that retreats.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class FoldStreakRiskConfig:
    """Threshold set for the fold-streak exposure throttle.

    Defaults target the audit-binding ``fold_negative_ic_streak <= 2``
    gate: ``kill_streak=3`` triggers the circuit breaker on the first
    streak that would breach the gate, so the dial cuts exposure
    *before* the eligibility metric crosses the threshold.

    All thresholds are operator-tunable. The dial's behavior must
    itself be governed by walk-forward evidence — these defaults are
    starting points, not certified.
    """

    #: Minimum number of completed prior folds before the dial activates.
    #: During warmup the scale is always 1.0 — we don't have enough OOS
    #: evidence to throttle on.
    min_folds_before_active: int = 4

    #: Hard circuit breaker. When the number of trailing consecutive
    #: negative-mean-IC folds reaches this value, the scale drops to
    #: 0.0 regardless of the EWMA. Recovers (scale > 0 again) only
    #: after a positive-IC fold breaks the streak.
    kill_streak: int = 3

    #: EWMA halflife in folds. ``4`` smooths over ~84 calendar days
    #: of test windows under the latest-stack 21-day fold cadence.
    ewma_halflife: int = 4

    #: EWMA IC at which the soft throttle bottoms out (scale = 0.0).
    floor_ic: float = -0.02

    #: EWMA IC at which the soft throttle tops out (scale = 1.0).
    ceiling_ic: float = 0.0

    def __post_init__(self) -> None:
        if self.min_folds_before_active < 1:
            raise ValueError("min_folds_before_active must be >= 1")
        if self.kill_streak < 1:
            raise ValueError("kill_streak must be >= 1")
        if self.ewma_halflife < 1:
            raise ValueError("ewma_halflife must be >= 1")
        if not math.isfinite(self.floor_ic) or not math.isfinite(self.ceiling_ic):
            raise ValueError("floor_ic and ceiling_ic must be finite")
        if self.floor_ic >= self.ceiling_ic:
            raise ValueError(
                "floor_ic must be < ceiling_ic so the linear interpolation has "
                f"positive support; got floor_ic={self.floor_ic}, "
                f"ceiling_ic={self.ceiling_ic}"
            )


StreakReason = Literal[
    "warmup",
    "circuit_breaker_kill",
    "ewma_floor",
    "ewma_ceiling",
    "ewma_throttled",
]


@dataclass(frozen=True)
class FoldStreakRiskScale:
    """Per-fold exposure throttle output + diagnostic context.

    Captures both the effective ``scale`` (the only number the
    evaluator needs) and the contributing signals so reviewers can
    see *why* the dial decided what it did each fold.
    """

    scale: float
    reason: StreakReason
    prior_fold_count: int
    neg_streak: int
    ewma_ic: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"scale must be in [0, 1]; got {self.scale}")
        if not math.isfinite(self.ewma_ic):
            raise ValueError(f"ewma_ic must be finite; got {self.ewma_ic}")

    def to_payload(self) -> dict[str, object]:
        return {
            "scale": float(self.scale),
            "reason": str(self.reason),
            "prior_fold_count": int(self.prior_fold_count),
            "neg_streak": int(self.neg_streak),
            "ewma_ic": float(self.ewma_ic),
        }


def compute_fold_streak_exposure_scale(
    prior_fold_ics: Sequence[float],
    *,
    config: FoldStreakRiskConfig,
) -> FoldStreakRiskScale:
    """Decide next-fold exposure scale from completed prior folds' ICs.

    ``prior_fold_ics`` must contain only fully-completed folds — the
    fold currently being evaluated must NOT appear in this sequence,
    otherwise its own OOS IC would leak into the exposure decision
    for that same fold (PIT violation).

    The returned scale is the product of two signals:
    * Hard circuit breaker: ``scale = 0`` when the trailing negative
      streak reaches ``config.kill_streak``.
    * Soft EWMA throttle: linear interpolation from ``floor_ic`` (0.0)
      to ``ceiling_ic`` (1.0). EWMAs above the ceiling cap at 1.0;
      EWMAs below the floor cap at 0.0.

    The effective scale is ``min(circuit_breaker, ewma_scale)`` so
    either signal can independently kill exposure.
    """
    prior_count = len(prior_fold_ics)

    # Warmup: no throttle until we have enough evidence to trust.
    if prior_count < config.min_folds_before_active:
        return FoldStreakRiskScale(
            scale=1.0,
            reason="warmup",
            prior_fold_count=prior_count,
            neg_streak=_trailing_negative_streak(prior_fold_ics),
            ewma_ic=0.0 if prior_count == 0 else _ewma(prior_fold_ics, config.ewma_halflife),
        )

    neg_streak = _trailing_negative_streak(prior_fold_ics)
    ewma_ic = _ewma(prior_fold_ics, config.ewma_halflife)

    if neg_streak >= config.kill_streak:
        return FoldStreakRiskScale(
            scale=0.0,
            reason="circuit_breaker_kill",
            prior_fold_count=prior_count,
            neg_streak=neg_streak,
            ewma_ic=ewma_ic,
        )

    if ewma_ic <= config.floor_ic:
        return FoldStreakRiskScale(
            scale=0.0,
            reason="ewma_floor",
            prior_fold_count=prior_count,
            neg_streak=neg_streak,
            ewma_ic=ewma_ic,
        )
    if ewma_ic >= config.ceiling_ic:
        return FoldStreakRiskScale(
            scale=1.0,
            reason="ewma_ceiling",
            prior_fold_count=prior_count,
            neg_streak=neg_streak,
            ewma_ic=ewma_ic,
        )

    # Linear interpolation: scale = (ewma - floor) / (ceiling - floor).
    span = config.ceiling_ic - config.floor_ic
    interpolated = (ewma_ic - config.floor_ic) / span
    # Clamp defensively in case of numerical edge cases.
    clamped = max(0.0, min(1.0, interpolated))
    return FoldStreakRiskScale(
        scale=clamped,
        reason="ewma_throttled",
        prior_fold_count=prior_count,
        neg_streak=neg_streak,
        ewma_ic=ewma_ic,
    )


def _trailing_negative_streak(values: Sequence[float]) -> int:
    """Length of the suffix of ``values`` consisting of strictly negative entries.

    A value of exactly 0.0 BREAKS the streak (treated as non-negative).
    NaN entries also break the streak — they convey "no signal", not
    "negative signal".
    """
    streak = 0
    for value in reversed(values):
        if math.isfinite(value) and value < 0.0:
            streak += 1
        else:
            break
    return streak


def _ewma(values: Sequence[float], halflife: int) -> float:
    """Exponentially-weighted mean with the most recent value weighted highest.

    Uses the canonical halflife → alpha conversion
    ``alpha = 1 - 0.5 ** (1 / halflife)``. Skips non-finite entries
    (they contribute neither weight nor value) so a missing-fold
    NaN doesn't poison the entire average.
    """
    if not values:
        return 0.0
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    weighted_sum = 0.0
    weight_total = 0.0
    # Walk oldest-to-newest, decaying each contribution.
    for i, value in enumerate(values):
        if not math.isfinite(value):
            continue
        age = len(values) - 1 - i
        weight = (1.0 - alpha) ** age
        weighted_sum += weight * value
        weight_total += weight
    if weight_total <= 0.0:
        return 0.0
    return weighted_sum / weight_total


def fold_streak_diagnostics_payload(
    config: FoldStreakRiskConfig | None,
    per_fold_scales: Sequence[FoldStreakRiskScale],
) -> dict[str, object]:
    """Aggregate per-fold streak diagnostics for evidence emission.

    Returns ``{"config": None, "applied": False}`` when the dial was
    not configured for this run, so dashboards can distinguish
    "dial off" from "dial on but never throttled".
    """
    if config is None:
        return {"config": None, "applied": False}
    scales = [s.scale for s in per_fold_scales]
    zero_count = sum(1 for s in scales if s == 0.0)
    scale_avg = sum(scales) / len(scales) if scales else 1.0
    scale_min = min(scales) if scales else 1.0
    return {
        "config": asdict(config),
        "applied": True,
        "n_folds": len(scales),
        "scale_avg": float(scale_avg),
        "scale_min": float(scale_min),
        "zero_fold_count": int(zero_count),
        "per_fold": [s.to_payload() for s in per_fold_scales],
    }


__all__ = [
    "FoldStreakRiskConfig",
    "FoldStreakRiskScale",
    "compute_fold_streak_exposure_scale",
    "fold_streak_diagnostics_payload",
]
