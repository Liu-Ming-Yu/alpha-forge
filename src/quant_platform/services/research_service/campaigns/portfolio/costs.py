"""Pluggable trading-cost models for the long-only portfolio evaluator.

Historically the evaluator priced a rebalance with a single flat assumption:
``cost = (slippage_bps / 1e4) * turnover``, where ``turnover`` is the
portfolio-aggregate sum of per-name ``|Δweight|``. That linear model is the
right first approximation for spread + commission, but it misses **market
impact** — the empirical fact that pushing more size through the book moves the
price against you, so cost grows *super*-linearly with trade size and a trade
concentrated in one name costs more than the same turnover spread across many.

This module factors the rebalance-pricing boundary into a small protocol so a
convex impact model can be swapped in without touching the construction logic
(targets, caps, no-trade band) or the return accounting. It mirrors the
``AlphaModel`` seam (``campaigns/models/``): a tiny, stateless object the
evaluator calls once per rebalance.

* :class:`LinearTurnoverCost` is the **behavior-preserving default** — a fold
  priced through it is bit-identical to the previous inlined
  ``turnover * bps / 1e4`` path, because it computes exactly that.
* :class:`QuadraticImpactCost` adds a per-name quadratic (Almgren-style)
  market-impact term on top of the linear spread/commission term — Arm K of the
  latest-stack backtest. See ADR-007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from quant_platform.core.constants import BPS_PER_UNIT

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping


@runtime_checkable
class TradingCostModel(Protocol):
    """Prices one rebalance from its per-name trade vector.

    ``trades`` maps instrument -> signed Δweight (today − yesterday). The model
    returns the cost as a **fraction of NAV** (return units), the same scale the
    evaluator subtracts from that day's gross return. A no-trade day passes an
    all-zero (or empty) vector and every reasonable model returns ``0.0``.
    """

    #: Stable identifier stamped into the evidence so an auditor can tell which
    #: cost assumption priced an arm. Must not encode anything machine-specific.
    name: str

    def cost(self, trades: Mapping[uuid.UUID, float]) -> float:
        """Return the execution cost (fraction of NAV) of ``trades``."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Self-describing parameters for the evidence/manifest audit trail."""
        ...


class LinearTurnoverCost:
    """``cost = (bps_per_turnover / 1e4) * Σ|Δw_i|`` — the behavior-preserving default.

    This is the exact arithmetic the evaluator used inline before the cost-model
    seam existed, so building one from a fold's ``slippage_bps_per_turnover`` and
    pricing through it reproduces the prior numbers bit-for-bit (the guard in
    ``test_trading_costs``). Spread + commission only; no market impact.
    """

    __slots__ = ("_bps", "name")

    def __init__(self, bps_per_turnover: float = 10.0) -> None:
        if bps_per_turnover < 0.0:
            raise ValueError("bps_per_turnover must be >= 0")
        self._bps = float(bps_per_turnover)
        self.name = f"linear-{self._bps:g}bps"

    def cost(self, trades: Mapping[uuid.UUID, float]) -> float:
        turnover = sum(abs(float(delta)) for delta in trades.values())
        return turnover * self._bps / BPS_PER_UNIT

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "type": "linear_turnover",
            "bps_per_turnover": self._bps,
        }


class QuadraticImpactCost:
    """Linear (spread/commission) + per-name quadratic (market-impact) cost.

    ``cost = (linear_bps / 1e4) * Σ|Δw_i|  +  quad_coef * Σ Δw_i²``

    The linear term is the same spread/commission the flat model charged. The
    quadratic term is the market impact the flat model ignored: it is computed
    **per name** (``Σ Δw_i²``, not ``(Σ|Δw_i|)²``), so for a fixed total turnover
    a trade concentrated in one name costs strictly more than the same turnover
    spread across many — the real execution incentive the flat model misses, and
    the convexity an Almgren temporary-impact model encodes.

    The quadratic coefficient is anchored to an interpretable point rather than
    given as a raw number: trading a single name at ``single_name_cap`` (a full
    position-cap-sized trade) incurs ``impact_bps_at_cap`` bps of *additional*
    impact on that name. Hence ``quad_coef = (impact_bps_at_cap / 1e4) /
    single_name_cap²``. The default (10 bps of impact at the 0.05 single-name
    cap) is a modeling assumption documented in ADR-007 with a sensitivity
    sweep; it is not calibrated to per-name ADV (volume is not plumbed into the
    weight-space evaluator), so it is a convex *robustness* model, not a
    venue-calibrated one.
    """

    __slots__ = ("_impact_bps_at_cap", "_linear_bps", "_quad_coef", "_single_name_cap", "name")

    def __init__(
        self,
        *,
        linear_bps_per_turnover: float = 10.0,
        impact_bps_at_cap: float = 10.0,
        single_name_cap: float = 0.05,
    ) -> None:
        if linear_bps_per_turnover < 0.0:
            raise ValueError("linear_bps_per_turnover must be >= 0")
        if impact_bps_at_cap < 0.0:
            raise ValueError("impact_bps_at_cap must be >= 0")
        if not 0.0 < single_name_cap <= 1.0:
            raise ValueError("single_name_cap must be in (0, 1]")
        self._linear_bps = float(linear_bps_per_turnover)
        self._impact_bps_at_cap = float(impact_bps_at_cap)
        self._single_name_cap = float(single_name_cap)
        # quad_coef has units of (return units) / (Δweight²); it converts the
        # per-name squared trade into a return-unit cost. Anchored so a Δw equal
        # to single_name_cap yields impact_bps_at_cap bps: quad_coef * cap² =
        # impact_bps_at_cap / 1e4.
        self._quad_coef = (self._impact_bps_at_cap / BPS_PER_UNIT) / (self._single_name_cap**2)
        self.name = (
            f"quad-impact-{self._linear_bps:g}lin"
            f"-{self._impact_bps_at_cap:g}imp@{self._single_name_cap:g}"
        )

    def cost(self, trades: Mapping[uuid.UUID, float]) -> float:
        linear = 0.0
        quadratic = 0.0
        for delta in trades.values():
            d = float(delta)
            linear += abs(d)
            quadratic += d * d
        return linear * self._linear_bps / BPS_PER_UNIT + self._quad_coef * quadratic

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "type": "quadratic_impact",
            "linear_bps_per_turnover": self._linear_bps,
            "impact_bps_at_cap": self._impact_bps_at_cap,
            "single_name_cap": self._single_name_cap,
            "quad_coef": self._quad_coef,
        }


__all__ = ["LinearTurnoverCost", "QuadraticImpactCost", "TradingCostModel"]
