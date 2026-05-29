"""Pluggable position-weighting schemes for the long-only target builder.

Historically `raw_long_only_target` assigned **equal weight** to every top-N
name (`investable_gross / N`, capped). Equal weight ignores risk: a 30%-vol
name and a 10%-vol name get the same dollar allocation, so the high-vol name
dominates portfolio risk. Inverse-volatility weighting (the qlib / risk-parity
move) tilts toward lower-risk names — weight ∝ 1/vol — which historically lifts
risk-adjusted return without changing the *selection* (the alpha still chooses
which names; weighting only sizes them).

This module factors the per-name sizing decision into a small protocol, mirroring
the `AlphaModel` (ADR-006) and `TradingCostModel` (ADR-007) seams. A scheme
returns **proportions** (summing to 1) over the selected names; the target
builder scales them by the same investable-gross budget equal weight used, so:

* :class:`EqualWeight` is the **behavior-preserving default** — `proportions`
  returns `1/N` for every name, so `raw_long_only_target` produces the exact
  weights it did before the seam existed (the guard in `test_weighting`).
* :class:`InverseVolWeight` sizes by shrunk inverse volatility — Arm L. See
  ADR-008.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@runtime_checkable
class WeightingScheme(Protocol):
    """Sizes the selected long-only names; selection is the alpha model's job.

    ``proportions`` receives the already-selected (and score-ordered) top-N rows
    and returns one non-negative proportion per row, in the same order, summing
    to 1. ``raw_long_only_target`` multiplies these by the investable-gross
    budget; the per-name and gross caps are enforced downstream, identically for
    every scheme.
    """

    #: Stable identifier stamped into the evidence audit trail.
    name: str

    def proportions(self, selected: Sequence[tuple[SupervisedAlphaSample, float]]) -> list[float]:
        """Return per-name proportions (>= 0, summing to 1) for ``selected``."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Self-describing parameters for the evidence/manifest audit trail."""
        ...


class EqualWeight:
    """``1/N`` for every selected name — the behavior-preserving default.

    With this scheme ``raw_long_only_target`` reproduces the prior equal-weight
    arithmetic exactly (``investable_gross / N``, capped), so arms that don't opt
    into a weighting scheme are bit-identical.
    """

    __slots__ = ("name",)

    def __init__(self) -> None:
        self.name = "equal-weight"

    def proportions(self, selected: Sequence[tuple[SupervisedAlphaSample, float]]) -> list[float]:
        n = len(selected)
        if n == 0:
            return []
        return [1.0 / n] * n

    def metadata(self) -> Mapping[str, object]:
        return {"name": self.name, "type": "equal"}


class InverseVolWeight:
    """Shrunk inverse-volatility weighting.

    For each selected name the realized volatility is read from a point-in-time
    feature (default ``low_vol_63d``, the price-volume family's sign-flipped 63d
    return std — so the magnitude is the vol). Weights are formed as:

        inv_i  = 1 / max(|vol_i|, vol_floor)
        p_i    = inv_i / Σ inv               (pure inverse-vol proportions)
        w_i    = shrinkage · (1/N) + (1 - shrinkage) · p_i

    ``shrinkage`` ∈ [0, 1] interpolates between **equal weight** (1.0 — identical
    to :class:`EqualWeight`) and **pure inverse-vol** (0.0). Shrinking toward
    equal weight blunts the estimation error in a single vol point estimate and
    keeps the book from concentrating in a handful of low-vol names — the same
    motivation as Ledoit-Wolf covariance shrinkage, applied to the weights.

    A name whose vol feature is missing or non-finite is assigned the
    cross-sectional **median** vol of the valid names (or ``vol_floor`` if none
    are valid), so a single bad feature value never silently drops a name or
    blows up its weight. The scheme never changes *which* names are held — only
    their relative size — so IC and the selected set are unchanged versus the
    equal-weight arm.
    """

    __slots__ = ("_shrinkage", "_vol_feature", "_vol_floor", "name")

    def __init__(
        self,
        *,
        vol_feature: str = "low_vol_63d",
        shrinkage: float = 0.5,
        vol_floor: float = 0.005,
    ) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if vol_floor <= 0.0:
            raise ValueError("vol_floor must be > 0")
        if not vol_feature:
            raise ValueError("vol_feature must be a non-empty feature name")
        self._vol_feature = vol_feature
        self._shrinkage = float(shrinkage)
        self._vol_floor = float(vol_floor)
        self.name = f"inverse-vol-{vol_feature}-s{shrinkage:g}"

    def _vols(self, selected: Sequence[tuple[SupervisedAlphaSample, float]]) -> list[float]:
        """Per-name vol magnitude; missing/non-finite -> median of valid (or floor)."""
        raw: list[float | None] = []
        for row, _ in selected:
            value = row.features.get(self._vol_feature)
            if value is None:
                raw.append(None)
                continue
            vol = abs(float(value))
            raw.append(vol if vol == vol and vol != float("inf") else None)  # NaN/inf -> None
        valid = sorted(v for v in raw if v is not None)
        if valid:
            mid = len(valid) // 2
            median = valid[mid] if len(valid) % 2 else (valid[mid - 1] + valid[mid]) / 2.0
        else:
            median = self._vol_floor
        return [v if v is not None else median for v in raw]

    def proportions(self, selected: Sequence[tuple[SupervisedAlphaSample, float]]) -> list[float]:
        n = len(selected)
        if n == 0:
            return []
        inv = [1.0 / max(vol, self._vol_floor) for vol in self._vols(selected)]
        total = sum(inv)
        # Degenerate guard: if every inverse is non-positive (impossible given the
        # floor, but defensive), fall back to equal weight.
        pure = [x / total for x in inv] if total > 0 else [1.0 / n] * n
        equal = 1.0 / n
        return [self._shrinkage * equal + (1.0 - self._shrinkage) * p for p in pure]

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "type": "inverse_vol",
            "vol_feature": self._vol_feature,
            "shrinkage": self._shrinkage,
            "vol_floor": self._vol_floor,
        }


__all__ = ["EqualWeight", "InverseVolWeight", "WeightingScheme"]
