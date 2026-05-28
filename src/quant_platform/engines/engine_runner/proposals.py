"""Proposal-generation behavior for EngineRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.engines.feature_jobs.dataset_guard import load_required_feature_dataset_id
from quant_platform.engines.feature_jobs.schema_guard import validate_required_feature_schema
from quant_platform.engines.proposals.builder import build_rejected_engine_target_proposal
from quant_platform.engines.proposals.events import publish_engine_proposal_generated
from quant_platform.engines.proposals.generation import generate_engine_target_proposal
from quant_platform.engines.session.public_api import hydrate_session_state

if TYPE_CHECKING:
    import asyncio
    import uuid
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.application.runtime.state import Session
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import EngineTargetProposal
    from quant_platform.core.domain.research.runs import StrategyRun
    from quant_platform.engines.framework.types import EngineConfig, EngineRunResult

log = structlog.get_logger(__name__)


class EngineRunnerProposalMixin:
    """V2 target-proposal generation methods."""

    _config: EngineConfig
    _settings: PlatformSettings
    _session: Session | None
    _strategy_run: StrategyRun | None
    _cycle_lock: asyncio.Lock
    _result: EngineRunResult

    if TYPE_CHECKING:

        async def _scheduled_feature_data(
            self,
            as_of: datetime,
        ) -> dict[uuid.UUID, dict[str, float]]: ...

    async def _generate_proposal_inner(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
        market_prices: dict[uuid.UUID, Decimal] | None,
        as_of: datetime,
        *,
        feature_dataset_id: uuid.UUID | None = None,
    ) -> EngineTargetProposal:
        session = self._session
        strategy_run = self._strategy_run
        if session is None or strategy_run is None:
            raise RuntimeError("call initialize() first")
        return await generate_engine_target_proposal(
            session=session,
            strategy_run=strategy_run,
            engine_name=self._config.engine_name,
            engine_version=self._config.engine_version,
            run_mode=self._config.run_mode,
            feature_data=feature_data,
            as_of=as_of,
            feature_dataset_id=feature_dataset_id,
        )

    async def generate_proposal(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]] | None = None,
        market_prices: dict[uuid.UUID, Decimal] | None = None,
        *,
        as_of: datetime | None = None,
    ) -> EngineTargetProposal:
        """Generate a target proposal without submitting orders."""
        session = self._session
        strategy_run = self._strategy_run
        if session is None or strategy_run is None:
            raise RuntimeError("call initialize() first")

        if self._cycle_lock.locked():
            log.warning(
                "engine_runner.proposal_reentrancy_guard",
                engine=self._config.engine_name,
                detail="previous cycle still running; skipping this tick",
            )
            now = as_of or session.clock.now()
            return build_rejected_engine_target_proposal(
                engine_name=self._config.engine_name,
                engine_version=self._config.engine_version,
                run_mode=self._config.run_mode,
                strategy_run_id=strategy_run.run_id,
                as_of=now,
                promotion_state="paper",
                note="reentrancy_guard_skipped",
            )

        async with self._cycle_lock:
            await hydrate_session_state(session)
            cycle_time = as_of or session.clock.now()
            effective_features = feature_data or await self._scheduled_feature_data(cycle_time)

            feature_dataset_id = await load_required_feature_dataset_id(
                session,
                settings=self._settings,
                feature_set_version=self._config.feature_set_name,
                as_of=cycle_time,
            )
            validate_required_feature_schema(
                engine_name=self._config.engine_name,
                feature_data=effective_features,
                required_features=self._config.required_features,
            )
            proposal = await self._generate_proposal_inner(
                effective_features,
                market_prices,
                cycle_time,
                feature_dataset_id=feature_dataset_id,
            )

            await publish_engine_proposal_generated(
                session.event_bus,
                proposal,
                occurred_at=cycle_time,
            )
            self._result.cycles_completed += 1
            return proposal
