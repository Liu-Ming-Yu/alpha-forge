"""Pure merge logic for account-level multi-engine targets."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    EngineBudget,
    EngineTargetContribution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.portfolio import PortfolioTarget

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MultiEngineMergeResult:
    combined_target: CombinedPortfolioTarget
    next_weights: dict[str, dict[uuid.UUID, Decimal]]


def merge_engine_targets(
    targets: Mapping[str, PortfolioTarget],
    *,
    budgets: tuple[EngineBudget, ...],
    previous_weights: Mapping[str, Mapping[uuid.UUID, Decimal]],
    as_of: datetime,
) -> MultiEngineMergeResult:
    """Merge engine targets under configured budgets and turnover caps."""
    combined_target_id = uuid.uuid4()
    engine_scaled: dict[str, dict[uuid.UUID, Decimal]] = {}
    contributions: list[EngineTargetContribution] = []
    notes: list[str] = []

    for budget in budgets:
        if not budget.enabled:
            continue
        target = targets.get(budget.engine_name)
        if target is None:
            notes.append(f"{budget.engine_name}: no target supplied")
            continue

        scaled_weights = _budget_scale_weights(target.weights, budget)
        scaled_weights = _cap_turnover(
            budget=budget,
            scaled_weights=scaled_weights,
            previous=previous_weights.get(budget.engine_name, {}),
            notes=notes,
        )

        engine_scaled[budget.engine_name] = scaled_weights
        contributions.append(
            EngineTargetContribution(
                contribution_id=uuid.uuid4(),
                combined_target_id=combined_target_id,
                engine_name=budget.engine_name,
                strategy_run_id=target.strategy_run_id,
                as_of=target.as_of,
                weights=scaled_weights,
                capital_weight=budget.capital_weight,
            )
        )

    contributions = _resolve_cross_engine_conflicts(
        budgets=budgets,
        engine_scaled=engine_scaled,
        contributions=contributions,
        combined_target_id=combined_target_id,
        notes=notes,
    )
    merged = _sum_engine_weights(engine_scaled)
    cleaned = {iid: w for iid, w in merged.items() if w > 0}
    invested = sum(cleaned.values(), Decimal("0"))
    if invested > Decimal("1"):
        raise ValueError(f"merged target gross exposure exceeds account: {invested}")

    return MultiEngineMergeResult(
        combined_target=CombinedPortfolioTarget(
            target_id=combined_target_id,
            as_of=as_of,
            weights=cleaned,
            cash_target_weight=Decimal("1") - invested,
            contributions=tuple(contributions),
            construction_notes=tuple(notes),
        ),
        next_weights={engine_name: dict(weights) for engine_name, weights in engine_scaled.items()},
    )


def _budget_scale_weights(
    weights: Mapping[uuid.UUID, Decimal],
    budget: EngineBudget,
) -> dict[uuid.UUID, Decimal]:
    gross_cap = min(budget.capital_weight, budget.max_gross)
    return {instrument_id: weight * gross_cap for instrument_id, weight in weights.items()}


_TURNOVER_PRUNE_EPSILON = Decimal("1E-9")


def _cap_turnover(
    *,
    budget: EngineBudget,
    scaled_weights: dict[uuid.UUID, Decimal],
    previous: Mapping[uuid.UUID, Decimal],
    notes: list[str],
) -> dict[uuid.UUID, Decimal]:
    # Turnover must be measured against `previous` for *every* instrument that
    # appears in either side, otherwise positions held in `previous` but dropped
    # from `scaled_weights` keep contributing their full delta to actual
    # turnover and the cap is silently violated.
    union = set(scaled_weights) | set(previous)
    deltas = {
        iid: scaled_weights.get(iid, Decimal("0")) - previous.get(iid, Decimal("0"))
        for iid in union
    }
    turnover = sum((abs(d) for d in deltas.values()), Decimal("0"))
    if turnover <= budget.max_turnover:
        return scaled_weights

    if budget.max_turnover == 0:
        # Zero-turnover budget: keep previous positions; nothing else satisfies
        # the constraint.
        notes.append(
            f"{budget.engine_name}: turnover {float(turnover):.4f} > 0 with "
            f"max_turnover=0; reverting to previous weights"
        )
        log.info(
            "multi_engine.turnover_locked",
            engine=budget.engine_name,
            raw_turnover=float(turnover),
        )
        capped_previous = {iid: previous.get(iid, Decimal("0")) for iid in union}
        return {iid: w for iid, w in capped_previous.items() if abs(w) > _TURNOVER_PRUNE_EPSILON}

    scale = budget.max_turnover / turnover
    # Scale the *deltas* and add `previous` back so the post-cap turnover is
    # actually `max_turnover` (not `current_turnover * scale`).
    capped = {iid: previous.get(iid, Decimal("0")) + deltas[iid] * scale for iid in union}
    capped = {iid: w for iid, w in capped.items() if abs(w) > _TURNOVER_PRUNE_EPSILON}
    notes.append(
        f"{budget.engine_name}: turnover {float(turnover):.4f} > "
        f"max {float(budget.max_turnover):.4f}; scaled by {float(scale):.4f}"
    )
    log.info(
        "multi_engine.turnover_capped",
        engine=budget.engine_name,
        raw_turnover=float(turnover),
        max_turnover=float(budget.max_turnover),
        scale=float(scale),
    )
    return capped


def _resolve_cross_engine_conflicts(
    *,
    budgets: tuple[EngineBudget, ...],
    engine_scaled: dict[str, dict[uuid.UUID, Decimal]],
    contributions: list[EngineTargetContribution],
    combined_target_id: uuid.UUID,
    notes: list[str],
) -> list[EngineTargetContribution]:
    budget_by_name = {budget.engine_name: budget for budget in budgets}
    engines_by_priority = sorted(
        engine_scaled,
        key=lambda engine_name: budget_by_name[engine_name].capital_weight,
        reverse=True,
    )
    instrument_allocated: dict[uuid.UUID, Decimal] = {}
    contribution_by_engine = {
        contribution.engine_name: contribution for contribution in contributions
    }

    for engine_name in engines_by_priority:
        scaled = engine_scaled[engine_name]
        budget = budget_by_name[engine_name]
        adjusted: dict[uuid.UUID, Decimal] = {}
        for instrument_id, weight in scaled.items():
            already = instrument_allocated.get(instrument_id, Decimal("0"))
            room = max(Decimal("0"), budget.max_gross - already)
            if weight > room:
                log.warning(
                    "multi_engine.instrument_conflict",
                    engine=engine_name,
                    instrument_id=str(instrument_id),
                    requested=float(weight),
                    room=float(room),
                )
                weight = room
                notes.append(
                    f"{engine_name}: instrument {instrument_id} reduced to {float(weight):.4f} "
                    f"(conflict with higher-priority engine)"
                )
            adjusted[instrument_id] = weight
            instrument_allocated[instrument_id] = already + weight
        engine_scaled[engine_name] = adjusted
        contribution = contribution_by_engine[engine_name]
        contribution_by_engine[engine_name] = EngineTargetContribution(
            contribution_id=contribution.contribution_id,
            combined_target_id=combined_target_id,
            engine_name=contribution.engine_name,
            strategy_run_id=contribution.strategy_run_id,
            as_of=contribution.as_of,
            weights=adjusted,
            capital_weight=contribution.capital_weight,
        )

    return [contribution_by_engine[contribution.engine_name] for contribution in contributions]


def _sum_engine_weights(
    engine_scaled: Mapping[str, Mapping[uuid.UUID, Decimal]],
) -> dict[uuid.UUID, Decimal]:
    merged: dict[uuid.UUID, Decimal] = {}
    for scaled in engine_scaled.values():
        for instrument_id, weight in scaled.items():
            merged[instrument_id] = merged.get(instrument_id, Decimal("0")) + weight
    return merged
