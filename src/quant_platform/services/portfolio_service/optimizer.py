"""Constraint-aware V2 portfolio optimizer."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio import (
    OptimizerResult,
    PortfolioRiskModel,
    PortfolioTarget,
    RiskLimits,
)
from quant_platform.services.portfolio_service.optimizer_math import (
    apply_factor_caps,
    covariance_scale,
    risk_snapshot,
    stress_cvar,
    stress_results,
)

if TYPE_CHECKING:
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot


class ConstraintAwareOptimizer:
    """Deterministic long-only optimizer for the V2 account orchestrator.

    This implementation intentionally avoids external solver dependencies.  It
    starts with engine-merged target weights, applies covariance-aware
    risk-scaling when variance terms exist, clips hard caps, rescales to gross
    and cash constraints, then computes factor and stress diagnostics.
    """

    def optimize(
        self,
        target: PortfolioTarget,
        account: AccountSnapshot,
        limits: RiskLimits,
        risk_model: PortfolioRiskModel,
    ) -> OptimizerResult:
        weights = {
            instrument_id: Decimal(str(weight)) for instrument_id, weight in target.weights.items()
        }
        notes: list[str] = list(target.construction_notes)
        binding: list[str] = []

        if not weights:
            snapshot = risk_snapshot(target, weights, risk_model, passed=True)
            return OptimizerResult(target=target, risk_snapshot=snapshot, binding_constraints=())

        adjusted = covariance_scale(
            weights,
            risk_model,
            halflife_days=limits.covariance_decay_halflife_days,
            target_as_of=target.as_of,
        )
        if adjusted != weights:
            binding.append("covariance_scaled")
            weights = adjusted

        cap = min(limits.max_single_name_weight, limits.max_gross_exposure)
        clipped: dict[uuid.UUID, Decimal] = {}
        for instrument_id, weight in weights.items():
            clipped_weight = max(Decimal("0"), min(weight, cap))
            if clipped_weight != weight:
                binding.append(f"single_name_cap:{instrument_id}")
            clipped[instrument_id] = clipped_weight
        weights = clipped

        max_invested = max(Decimal("0"), limits.max_gross_exposure - limits.min_cash_buffer)
        gross = sum(weights.values(), Decimal("0"))
        if gross > max_invested and gross > Decimal("0"):
            scale = max_invested / gross
            weights = {instrument_id: weight * scale for instrument_id, weight in weights.items()}
            binding.append("gross_or_cash_cap")
            gross = sum(weights.values(), Decimal("0"))

        capped = apply_factor_caps(weights, risk_model, limits.max_sector_weight)
        if capped != weights:
            binding.append("factor_cap")
            weights = capped
            gross = sum(weights.values(), Decimal("0"))

        if account.net_asset_value <= Decimal("0"):
            weights = {}
            gross = Decimal("0")
            binding.append("zero_nav")

        stress = stress_results(weights, risk_model)
        cvar = stress_cvar(stress, limits.cvar_tail_pct)
        cvar_limit = abs(limits.max_drawdown_halt)
        passed = cvar is None or cvar <= cvar_limit
        if not passed:
            binding.append("stress_cvar_halt")
            weights = {}
            gross = Decimal("0")

        cash = Decimal("1") - gross
        optimized = PortfolioTarget(
            target_id=uuid.uuid4(),
            strategy_run_id=target.strategy_run_id,
            as_of=target.as_of,
            regime_id=target.regime_id,
            weights={
                instrument_id: weight for instrument_id, weight in weights.items() if weight > 0
            },
            cash_target_weight=cash,
            construction_notes=notes + binding,
        )
        snapshot = risk_snapshot(optimized, optimized.weights, risk_model, passed=passed)
        return OptimizerResult(
            target=optimized,
            risk_snapshot=snapshot,
            binding_constraints=tuple(binding),
        )
