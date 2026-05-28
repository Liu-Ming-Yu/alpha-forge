"""Unit tests for engine proposal DTO and event helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.framework.types import RunMode
from quant_platform.engines.proposals.builder import (
    build_engine_target_proposal,
    build_rejected_engine_target_proposal,
)
from quant_platform.engines.proposals.events import (
    engine_proposal_generated_event,
    publish_engine_proposal_generated,
)

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


def test_rejected_proposal_defaults_live_mode_to_live_promotion_state() -> None:
    strategy_run_id = uuid.uuid4()

    proposal = build_rejected_engine_target_proposal(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.LIVE,
        strategy_run_id=strategy_run_id,
        as_of=_AS_OF,
        note="risk_policy_rejected",
    )

    assert proposal.engine_name == "equity"
    assert proposal.run_mode == "live"
    assert proposal.strategy_run_id == strategy_run_id
    assert proposal.weights == {}
    assert proposal.cash_target_weight == Decimal("1")
    assert proposal.promotion_state == "live"
    assert proposal.notes == ("risk_policy_rejected",)


def test_rejected_proposal_can_force_shadow_safe_paper_state() -> None:
    proposal = build_rejected_engine_target_proposal(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.LIVE,
        strategy_run_id=uuid.uuid4(),
        as_of=_AS_OF,
        promotion_state="paper",
        note="reentrancy_guard_skipped",
    )

    assert proposal.run_mode == "live"
    assert proposal.promotion_state == "paper"
    assert proposal.notes == ("reentrancy_guard_skipped",)


def test_target_proposal_preserves_target_payload() -> None:
    instrument_id = uuid.uuid4()
    feature_dataset_id = uuid.uuid4()

    proposal = build_engine_target_proposal(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.PAPER,
        strategy_run_id=uuid.uuid4(),
        as_of=_AS_OF,
        weights={instrument_id: Decimal("0.25")},
        cash_target_weight=Decimal("0.75"),
        construction_notes=["sector_cap_applied", "turnover_ok"],
        feature_dataset_id=feature_dataset_id,
    )

    assert proposal.run_mode == "paper"
    assert proposal.weights == {instrument_id: Decimal("0.25")}
    assert proposal.cash_target_weight == Decimal("0.75")
    assert proposal.promotion_state == "paper"
    assert proposal.feature_dataset_id == feature_dataset_id
    assert proposal.notes == ("sector_cap_applied", "turnover_ok")


def test_proposal_generated_event_counts_weights() -> None:
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

    event = engine_proposal_generated_event(proposal, occurred_at=_AS_OF)

    assert event.proposal_id == proposal.proposal_id
    assert event.engine_name == "equity"
    assert event.as_of == _AS_OF
    assert event.weight_count == 1


@pytest.mark.asyncio
async def test_publish_proposal_generated_event_uses_standard_event_shape() -> None:
    proposal = build_rejected_engine_target_proposal(
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.PAPER,
        strategy_run_id=uuid.uuid4(),
        as_of=_AS_OF,
        note="risk_policy_rejected",
    )
    event_bus = AsyncMock()

    await publish_engine_proposal_generated(event_bus, proposal, occurred_at=_AS_OF)

    event_bus.publish.assert_awaited_once()
    event = event_bus.publish.await_args.args[0]
    assert event.proposal_id == proposal.proposal_id
    assert event.weight_count == 0
