"""Domain-event helpers for engine target proposals."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.events import EngineProposalGenerated

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.domain.production import EngineTargetProposal


class ProposalEventBus(Protocol):
    async def publish(self, event: EngineProposalGenerated) -> None:
        """Publish a proposal event."""


def engine_proposal_generated_event(
    proposal: EngineTargetProposal,
    *,
    occurred_at: datetime,
) -> EngineProposalGenerated:
    """Build the event emitted after an engine target proposal is generated."""
    return EngineProposalGenerated(
        event_id=uuid.uuid4(),
        occurred_at=occurred_at,
        proposal_id=proposal.proposal_id,
        engine_name=proposal.engine_name,
        as_of=proposal.as_of,
        weight_count=len(proposal.weights),
    )


async def publish_engine_proposal_generated(
    event_bus: ProposalEventBus,
    proposal: EngineTargetProposal,
    *,
    occurred_at: datetime,
) -> None:
    """Publish the standard proposal-generated event."""
    await event_bus.publish(engine_proposal_generated_event(proposal, occurred_at=occurred_at))
