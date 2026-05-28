"""Proposal admission helpers for multi-engine account orchestration."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio import PortfolioTarget

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.core.domain.production import EngineBudget, EngineTargetProposal


def build_proposal_targets(
    proposals: Iterable[EngineTargetProposal],
    *,
    budgets: tuple[EngineBudget, ...],
    v2_enabled: bool,
    account_orchestrator_enabled: bool,
    require_feature_datasets: bool,
    require_promotion_gate: bool,
) -> dict[str, PortfolioTarget]:
    """Validate engine proposals and convert them into mergeable targets."""
    proposal_targets: dict[str, PortfolioTarget] = {}
    budgets_by_engine = {budget.engine_name: budget for budget in budgets}
    for proposal in proposals:
        if proposal.engine_name in proposal_targets:
            raise ValueError(f"duplicate proposal for engine {proposal.engine_name}")
        budget = budgets_by_engine.get(proposal.engine_name)
        if budget is None:
            raise ValueError(f"no capital budget configured for {proposal.engine_name}")
        _assert_proposal_admitted(
            proposal,
            budget=budget,
            v2_enabled=v2_enabled,
            account_orchestrator_enabled=account_orchestrator_enabled,
            require_feature_datasets=require_feature_datasets,
            require_promotion_gate=require_promotion_gate,
        )
        proposal_targets[proposal.engine_name] = PortfolioTarget(
            target_id=proposal.proposal_id,
            strategy_run_id=proposal.strategy_run_id,
            as_of=proposal.as_of,
            regime_id=uuid.uuid5(uuid.NAMESPACE_URL, str(proposal.proposal_id)),
            weights=proposal.weights,
            cash_target_weight=proposal.cash_target_weight,
            construction_notes=list(proposal.notes),
        )
    return proposal_targets


def _assert_proposal_admitted(
    proposal: EngineTargetProposal,
    *,
    budget: EngineBudget,
    v2_enabled: bool,
    account_orchestrator_enabled: bool,
    require_feature_datasets: bool,
    require_promotion_gate: bool,
) -> None:
    if proposal.engine_version != budget.engine_version:
        raise ValueError(
            f"{proposal.engine_name} version mismatch: proposal "
            f"{proposal.engine_version} != budget {budget.engine_version}"
        )
    if proposal.run_mode == "live" and not (v2_enabled and account_orchestrator_enabled):
        raise RuntimeError("live engine proposals require V2 account orchestrator")
    if require_feature_datasets and proposal.feature_dataset_id is None:
        raise RuntimeError(f"{proposal.engine_name} missing feature_dataset_id under V2 live gates")
    if (
        require_promotion_gate
        and proposal.run_mode == "live"
        and (proposal.promotion_state != "live" or proposal.model_artifact_id is None)
    ):
        raise RuntimeError(f"{proposal.engine_name} live proposal lacks promoted model artifact")
