"""V2 proposal-only cycle helper for engine runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.runtime.state import CycleResult
from quant_platform.engines.proposals.events import (
    ProposalEventBus,
    publish_engine_proposal_generated,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from quant_platform.core.domain.production import EngineTargetProposal


async def run_v2_proposal_cycle(
    *,
    proposal_factory: Callable[[], Awaitable[EngineTargetProposal]],
    event_bus: ProposalEventBus,
    occurred_at: datetime,
) -> CycleResult:
    """Generate, publish, and wrap an engine proposal without submitting orders."""
    proposal = await proposal_factory()
    await publish_engine_proposal_generated(
        event_bus,
        proposal,
        occurred_at=occurred_at,
    )
    return CycleResult(
        signals=[],
        target=None,
        approved=[],
        rejected=[],
        submitted_ids=[],
        fills=[],
        proposal=proposal,
    )
