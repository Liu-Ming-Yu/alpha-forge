from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderStateEventType,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.production import (
    CombinedPortfolioTarget,
    EngineTargetContribution,
)
from quant_platform.engines.account.orchestrator_mapping import (
    UNKNOWN_PARENT_PROPOSAL_ID,
    combined_to_portfolio_target,
    order_allocations_for_intents,
    parent_proposal_ids_for_orders,
)
from quant_platform.engines.account.order_lifecycle import (
    append_acknowledged_events,
    append_created_events,
)
from quant_platform.infrastructure.v2.state import InMemoryOrderStateStore

AS_OF = datetime(2026, 1, 2, tzinfo=UTC)


def _intent(instrument_id: uuid.UUID, *, quantity: int = 3) -> OrderIntent:
    return OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.MOC,
        time_in_force=TimeInForce.DAY,
        created_at=AS_OF,
    )


def _combined(instrument_id: uuid.UUID, contribution_id: uuid.UUID) -> CombinedPortfolioTarget:
    return CombinedPortfolioTarget(
        target_id=uuid.uuid4(),
        as_of=AS_OF,
        weights={instrument_id: Decimal("0.25")},
        cash_target_weight=Decimal("0.75"),
        contributions=(
            EngineTargetContribution(
                contribution_id=contribution_id,
                combined_target_id=uuid.uuid4(),
                engine_name="cross_sectional_equity_v1",
                strategy_run_id=uuid.uuid4(),
                as_of=AS_OF,
                weights={instrument_id: Decimal("0.25")},
                capital_weight=Decimal("0.70"),
            ),
        ),
        construction_notes=("scaled",),
    )


def test_combined_to_portfolio_target_preserves_optimizer_contract_fields() -> None:
    strategy_run_id = uuid.uuid4()
    instrument_id = uuid.uuid4()
    combined = _combined(instrument_id, uuid.uuid4())

    target = combined_to_portfolio_target(combined, strategy_run_id=strategy_run_id)

    assert target.target_id == combined.target_id
    assert target.strategy_run_id == strategy_run_id
    assert target.weights == combined.weights
    assert target.cash_target_weight == Decimal("0.75")
    assert tuple(target.construction_notes) == ("scaled",)
    assert target.regime_id == uuid.uuid5(uuid.NAMESPACE_URL, str(combined.target_id))


def test_parent_proposal_ids_for_orders_is_parallel_to_planned_orders() -> None:
    known_instrument = uuid.uuid4()
    unknown_instrument = uuid.uuid4()
    contribution_id = uuid.uuid4()
    combined = _combined(known_instrument, contribution_id)
    planned = (_intent(known_instrument), _intent(unknown_instrument))

    parent_ids = parent_proposal_ids_for_orders(planned, combined)

    assert parent_ids == (contribution_id, UNKNOWN_PARENT_PROPOSAL_ID)


def test_order_allocations_for_intents_attributes_positive_contributions() -> None:
    instrument_id = uuid.uuid4()
    combined = _combined(instrument_id, uuid.uuid4())
    intent = _intent(instrument_id, quantity=7)
    allocation_id = uuid.uuid4()

    allocations = order_allocations_for_intents(
        [intent],
        combined,
        {instrument_id: Decimal("11.50")},
        allocation_id_factory=lambda: allocation_id,
    )

    assert len(allocations) == 1
    assert allocations[0].allocation_id == allocation_id
    assert allocations[0].order_id == intent.order_id
    assert allocations[0].allocated_weight == Decimal("0.25")
    assert allocations[0].allocated_notional == Decimal("80.50")


@pytest.mark.asyncio
async def test_acknowledged_event_helper_is_idempotent_when_already_acknowledged() -> None:
    order_state = InMemoryOrderStateStore()
    intent = _intent(uuid.uuid4())

    await append_created_events(order_state, [intent], AS_OF)
    await append_acknowledged_events(order_state, [intent], [intent.order_id], AS_OF)
    await append_acknowledged_events(order_state, [intent], [intent.order_id], AS_OF)

    events = await order_state.list_events(intent.order_id)
    assert [event.event_type for event in events] == [
        OrderStateEventType.CREATED,
        OrderStateEventType.ACKNOWLEDGED,
    ]
    assert events[-1].status == OrderStatus.SUBMITTED
