"""Account-level multi-engine coordination primitives.

The current production-safe boundary is deliberate: multiple engines may be
merged in shadow/paper, but concurrent live execution is blocked until this
runner owns the single account-level order submission pass.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.config import PlatformSettings
from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    EngineBudget,
    EngineTargetProposal,
)
from quant_platform.engines.multi_engine.merge import merge_engine_targets
from quant_platform.engines.multi_engine.proposals import build_proposal_targets
from quant_platform.infrastructure.repositories.multi_engine_governance import (
    build_multi_engine_governance_repository,
)
from quant_platform.infrastructure.support.clock import WallClock
from quant_platform.infrastructure.support.distributed_lock import create_distributed_lock

if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator, Iterable, Mapping
    from datetime import datetime

    from quant_platform.core.contracts import Clock, MultiEngineGovernanceRepository
    from quant_platform.core.domain.portfolio import PortfolioTarget

DEFAULT_ENGINE_BUDGETS: tuple[EngineBudget, ...] = (
    EngineBudget(
        engine_name="cross_sectional_equity_v1",
        engine_version="0.1.0",
        run_mode="paper",
        capital_weight=Decimal("0.70"),
        max_gross=Decimal("0.70"),
        max_turnover=Decimal("0.20"),
    ),
    EngineBudget(
        engine_name="etf_macro_allocator_v1",
        engine_version="0.1.0",
        run_mode="paper",
        capital_weight=Decimal("0.25"),
        max_gross=Decimal("0.25"),
        max_turnover=Decimal("0.10"),
    ),
    EngineBudget(
        engine_name="event_news_overlay_v1",
        engine_version="0.1.0",
        run_mode="shadow",
        capital_weight=Decimal("0.05"),
        max_gross=Decimal("0.05"),
        max_turnover=Decimal("0.05"),
        enabled=False,
    ),
)


class MultiEngineRunner:
    """Merge engine targets under one account-level budget.

    Engines submit proposal-only targets.  This runner owns the single
    account-level merge boundary and is the only live-safe entry point for
    multi-engine targets.
    """

    def __init__(
        self,
        settings: PlatformSettings | None = None,
        *,
        account_id: str = "",
        budgets: tuple[EngineBudget, ...] = DEFAULT_ENGINE_BUDGETS,
        governance_repo: MultiEngineGovernanceRepository | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings or PlatformSettings()
        self._account_id = account_id or self._settings.broker.account_id or "default"
        self._budgets = tuple(budgets)
        self._clock = clock or WallClock()
        self._governance = governance_repo or build_multi_engine_governance_repository(
            self._settings.storage.postgres_dsn
        )
        # Previous cycle weights per engine, keyed by engine_name -> {instrument_id: weight}.
        # Used to compute per-engine turnover for max_turnover enforcement.
        self._prev_weights: dict[str, dict[uuid.UUID, Decimal]] = {}
        self._assert_live_safe()

    @property
    def budgets(self) -> tuple[EngineBudget, ...]:
        return self._budgets

    def _assert_live_safe(self) -> None:
        live_enabled = [
            budget.engine_name
            for budget in self._budgets
            if budget.enabled and budget.run_mode == "live"
        ]
        if len(live_enabled) > 1 and not (
            self._settings.v2.enabled and self._settings.v2.account_orchestrator_enabled
        ):
            raise RuntimeError(
                "Concurrent multi-engine live execution requires V2 account "
                "orchestrator ownership. Set QP__V2__ENABLED=true and "
                "QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true."
            )

        # Budget invariants: catch misconfiguration before a single order ships.
        enabled = [b for b in self._budgets if b.enabled]
        total_weight = sum(b.capital_weight for b in enabled)
        if total_weight > Decimal("1.0"):
            raise ValueError(
                f"Total capital_weight {total_weight} across enabled engines exceeds 100%. "
                f"Engines: {[b.engine_name for b in enabled]}"
            )
        for b in self._budgets:
            if not b.enabled:
                continue
            if b.max_gross > b.capital_weight:
                raise ValueError(
                    f"Engine {b.engine_name}: max_gross {b.max_gross} exceeds "
                    f"capital_weight {b.capital_weight}."
                )
            if b.max_turnover <= Decimal("0"):
                raise ValueError(
                    f"Engine {b.engine_name}: max_turnover must be > 0, got {b.max_turnover}."
                )

    async def persist_budgets(self) -> None:
        for budget in self._budgets:
            await self._governance.save_engine_budget(budget)

    async def merge_targets(
        self,
        targets: Mapping[str, PortfolioTarget],
        *,
        as_of: datetime | None = None,
    ) -> CombinedPortfolioTarget:
        """Merge engine targets by budget-scaling each engine's weights."""
        async with self._account_cycle_lock():
            target = self._merge_targets_unlocked(targets, as_of=as_of)
            await self.persist_budgets()
            await self._governance.save_combined_target(target)
            return target

    async def merge_proposals(
        self,
        proposals: Iterable[EngineTargetProposal],
        *,
        as_of: datetime | None = None,
    ) -> CombinedPortfolioTarget:
        """Merge V2 proposal-only engine outputs under central account control."""
        proposal_targets = build_proposal_targets(
            proposals,
            budgets=self._budgets,
            v2_enabled=self._settings.v2.enabled,
            account_orchestrator_enabled=self._settings.v2.account_orchestrator_enabled,
            require_feature_datasets=self._settings.v2.require_feature_datasets,
            require_promotion_gate=self._settings.alpha.require_promotion_gate,
        )
        return await self.merge_targets(proposal_targets, as_of=as_of)

    def _merge_targets_unlocked(
        self,
        targets: Mapping[str, PortfolioTarget],
        *,
        as_of: datetime | None = None,
    ) -> CombinedPortfolioTarget:
        result = merge_engine_targets(
            targets,
            budgets=self._budgets,
            previous_weights=self._prev_weights,
            as_of=as_of or self._clock.now(),
        )
        self._prev_weights.update(result.next_weights)
        return result.combined_target

    @asynccontextmanager
    async def _account_cycle_lock(self) -> AsyncIterator[object]:
        lock = create_distributed_lock(
            self._settings.storage.redis_url,
            f"account_cycle:{self._account_id}",
            ttl_seconds=self._settings.storage.distributed_lock_ttl_seconds,
            acquire_timeout_seconds=self._settings.storage.distributed_lock_acquire_timeout_seconds,
            renew_interval_seconds=self._settings.storage.distributed_lock_renew_interval_seconds,
        )
        async with lock:
            yield lock
