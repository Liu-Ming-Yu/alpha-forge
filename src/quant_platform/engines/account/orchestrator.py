"""V2 account-level execution orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderStateEventType,
    OrderStatus,
)
from quant_platform.engines.account.orchestrator_mapping import (
    combined_to_portfolio_target,
    order_allocations_for_intents,
    parent_proposal_ids_for_orders,
)
from quant_platform.engines.account.order_lifecycle import (
    append_acknowledged_events,
    append_approval_events,
    append_created_events,
    append_order_state,
    route_event_payload,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.contracts import (
        ExecutionRouter,
        MultiEngineGovernanceRepository,
        Optimizer,
        OrderStateStore,
        PortfolioRiskModelRepository,
    )
    from quant_platform.core.domain.portfolio import PortfolioRiskModel, PortfolioTarget, RiskLimits
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.production import (
        CombinedPortfolioTarget,
        EngineTargetProposal,
    )
    from quant_platform.engines.multi_engine import MultiEngineRunner
    from quant_platform.services.execution_service.orders.controllers import (
        SubmitOrdersControllerImpl,
    )
    from quant_platform.services.portfolio_service.controllers import ApproveOrdersControllerImpl
    from quant_platform.services.portfolio_service.order_planner import PortfolioTargetOrderPlanner

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AccountExecutionResult:
    """Summary of one account-level V2 execution pass."""

    combined_target: CombinedPortfolioTarget
    optimized_target: PortfolioTarget
    planned: tuple[OrderIntent, ...]
    approved: tuple[OrderIntent, ...]
    rejected: tuple[OrderIntent, ...]
    submitted_ids: tuple[uuid.UUID, ...]
    parent_proposal_ids: tuple[
        uuid.UUID, ...
    ] = ()  # parallel to planned; source proposal per order


class AccountExecutionOrchestrator:
    """Only V2 component allowed to submit multi-engine live orders."""

    def __init__(
        self,
        *,
        multi_engine: MultiEngineRunner,
        optimizer: Optimizer,
        order_planner: PortfolioTargetOrderPlanner,
        approve_ctrl: ApproveOrdersControllerImpl,
        submit_ctrl: SubmitOrdersControllerImpl,
        order_state: OrderStateStore,
        execution_router: ExecutionRouter,
        risk_repo: PortfolioRiskModelRepository | None = None,
        governance_repo: MultiEngineGovernanceRepository | None = None,
    ) -> None:
        self._multi_engine = multi_engine
        self._optimizer = optimizer
        self._order_planner = order_planner
        self._approve = approve_ctrl
        self._submit = submit_ctrl
        self._order_state = order_state
        self._router = execution_router
        self._risk_repo = risk_repo
        self._governance = governance_repo

    async def execute(
        self,
        *,
        proposals: tuple[EngineTargetProposal, ...],
        account: AccountSnapshot,
        limits: RiskLimits,
        risk_model: PortfolioRiskModel,
        market_prices: dict[uuid.UUID, Decimal],
        strategy_run_id: uuid.UUID,
        as_of: datetime,
    ) -> AccountExecutionResult:
        combined = await self._multi_engine.merge_proposals(proposals, as_of=as_of)

        target = combined_to_portfolio_target(combined, strategy_run_id=strategy_run_id)
        optimized = self._optimizer.optimize(target, account, limits, risk_model)
        if self._risk_repo is not None:
            await self._risk_repo.save_risk_snapshot(optimized.risk_snapshot)

        planned = self._order_planner.plan(
            optimized.target,
            account,
            market_prices,
            strategy_run_id,
        )

        parent_proposal_ids = parent_proposal_ids_for_orders(planned, combined)
        await append_created_events(self._order_state, planned, as_of)

        approved, rejected = await self._approve.approve(planned, account)
        await append_approval_events(self._order_state, approved, rejected, as_of)

        if not approved:
            log.info(
                "account_orchestrator.no_approved_orders",
                planned_count=len(planned),
                rejected_count=len(rejected),
            )
            return AccountExecutionResult(
                combined_target=combined,
                optimized_target=optimized.target,
                planned=tuple(planned),
                approved=(),
                rejected=tuple(rejected),
                submitted_ids=(),
                parent_proposal_ids=parent_proposal_ids,
            )

        if self._governance is not None:
            await self._save_allocations(approved, combined, market_prices)

        for intent in approved:
            route = self._router.route(intent)
            await append_order_state(
                self._order_state,
                intent,
                OrderStateEventType.ROUTED,
                as_of,
                OrderStatus.APPROVED,
                payload=route_event_payload(route),
            )

        submitted_ids = await self._submit.submit(approved, account=account)
        await append_acknowledged_events(self._order_state, approved, submitted_ids, as_of)

        return AccountExecutionResult(
            combined_target=combined,
            optimized_target=optimized.target,
            planned=tuple(planned),
            approved=tuple(approved),
            rejected=tuple(rejected),
            submitted_ids=tuple(submitted_ids),
            parent_proposal_ids=parent_proposal_ids,
        )

    async def _save_allocations(
        self,
        intents: list[OrderIntent],
        target: CombinedPortfolioTarget,
        market_prices: dict[uuid.UUID, Decimal],
    ) -> None:
        if self._governance is None:
            return
        allocations = order_allocations_for_intents(intents, target, market_prices)
        if allocations:
            await self._governance.save_order_allocations(allocations)
