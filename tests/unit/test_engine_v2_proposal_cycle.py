"""Unit tests for V2 proposal-only cycle helper."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.framework.types import RunMode
from quant_platform.engines.proposals.builder import build_engine_target_proposal
from quant_platform.engines.proposals.v2_cycle import run_v2_proposal_cycle

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_v2_proposal_cycle_publishes_and_returns_proposal_result() -> None:
    instrument_id = uuid.uuid4()
    proposal = build_engine_target_proposal(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.PAPER,
        strategy_run_id=uuid.uuid4(),
        as_of=_AS_OF,
        weights={instrument_id: Decimal("0.25")},
        cash_target_weight=Decimal("0.75"),
        construction_notes=[],
    )
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()

    async def _proposal_factory():
        return proposal

    result = await run_v2_proposal_cycle(
        proposal_factory=_proposal_factory,
        event_bus=event_bus,
        occurred_at=_AS_OF,
    )

    assert result.proposal is proposal
    assert result.signals == []
    assert result.target is None
    assert result.submitted_ids == []
    event_bus.publish.assert_awaited_once()
    event = event_bus.publish.await_args.args[0]
    assert event.proposal_id == proposal.proposal_id
    assert event.weight_count == 1
