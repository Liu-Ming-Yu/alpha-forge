"""Generate engine target proposals from the shared target pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.engines.proposals.builder import (
    build_engine_target_proposal,
    build_rejected_engine_target_proposal,
)
from quant_platform.engines.proposals.target_pipeline import build_engine_target

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.production import EngineTargetProposal
    from quant_platform.core.domain.research.runs import StrategyRun


async def generate_engine_target_proposal(
    *,
    session: Session,
    strategy_run: StrategyRun,
    engine_name: str,
    engine_version: str,
    run_mode: object,
    feature_data: dict[uuid.UUID, dict[str, float]],
    as_of: datetime,
    feature_dataset_id: uuid.UUID | None = None,
) -> EngineTargetProposal:
    """Generate a proposal DTO from current feature data and engine config."""
    target_build = await build_engine_target(
        session=session,
        strategy_run=strategy_run,
        feature_data=feature_data,
        as_of=as_of,
    )

    if target_build.target is None:
        return build_rejected_engine_target_proposal(
            engine_name=engine_name,
            engine_version=engine_version,
            run_mode=run_mode,
            strategy_run_id=strategy_run.run_id,
            as_of=as_of,
            note="risk_policy_rejected",
        )

    return build_engine_target_proposal(
        engine_name=engine_name,
        engine_version=engine_version,
        run_mode=run_mode,
        strategy_run_id=strategy_run.run_id,
        as_of=target_build.target.as_of,
        weights=target_build.target.weights,
        cash_target_weight=target_build.target.cash_target_weight,
        feature_dataset_id=feature_dataset_id,
        construction_notes=tuple(target_build.target.construction_notes),
    )
