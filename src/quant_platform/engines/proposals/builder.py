"""Construct target proposal DTOs emitted by engine runners."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import EngineTargetProposal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime


def promotion_state_for_run_mode(run_mode: object) -> str:
    """Map an engine run mode to the proposal promotion state."""
    return "live" if getattr(run_mode, "value", run_mode) == "live" else "paper"


def build_rejected_engine_target_proposal(
    *,
    engine_name: str,
    engine_version: str,
    run_mode: object,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    note: str,
    promotion_state: str | None = None,
) -> EngineTargetProposal:
    """Build an all-cash proposal for runner guards and risk-policy rejection."""
    run_mode_value = getattr(run_mode, "value", run_mode)
    return EngineTargetProposal(
        proposal_id=uuid.uuid4(),
        engine_name=engine_name,
        engine_version=engine_version,
        run_mode=str(run_mode_value),
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        weights={},
        cash_target_weight=Decimal("1"),
        promotion_state=promotion_state or promotion_state_for_run_mode(run_mode),
        notes=(note,),
    )


def build_engine_target_proposal(
    *,
    engine_name: str,
    engine_version: str,
    run_mode: object,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    weights: Mapping[uuid.UUID, Decimal],
    cash_target_weight: Decimal,
    construction_notes: Sequence[str],
    feature_dataset_id: uuid.UUID | None = None,
) -> EngineTargetProposal:
    """Build a proposal from a portfolio target."""
    run_mode_value = getattr(run_mode, "value", run_mode)
    return EngineTargetProposal(
        proposal_id=uuid.uuid4(),
        engine_name=engine_name,
        engine_version=engine_version,
        run_mode=str(run_mode_value),
        strategy_run_id=strategy_run_id,
        as_of=as_of,
        weights=dict(weights),
        cash_target_weight=cash_target_weight,
        promotion_state=promotion_state_for_run_mode(run_mode),
        feature_dataset_id=feature_dataset_id,
        notes=tuple(construction_notes),
    )
