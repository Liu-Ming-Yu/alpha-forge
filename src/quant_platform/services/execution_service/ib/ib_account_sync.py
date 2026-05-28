"""IB account and position sync helpers."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.core.exceptions import BrokerUnavailableError
from quant_platform.services.execution_service.ib.ib_account_mapper import (
    account_snapshot_from_values,
    position_snapshot_from_values,
)
from quant_platform.services.execution_service.ib.ib_contract_mapper import contract_con_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot

log = structlog.get_logger(__name__)


async def sync_account_snapshot(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    sync_positions: Callable[[], Awaitable[Sequence[PositionSnapshot]]],
) -> AccountSnapshot:
    """Run the IB account-summary request and return a domain account snapshot."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    loop = asyncio.get_running_loop()
    req_id = 9001
    wrapper_any._account_values = {}
    wrapper_any._account_active_req_id = req_id
    wrapper_any._account_done = loop.create_future()

    client_any.reqAccountSummary(
        req_id,
        "All",
        "NetLiquidation,TotalCashValue,SettledCash,BuyingPower",
    )

    try:
        values = await asyncio.wait_for(wrapper_any._account_done, timeout=timeout)
    except TimeoutError as exc:
        wrapper_any._account_active_req_id = None
        client_any.cancelAccountSummary(req_id)
        raise BrokerUnavailableError("sync_account timed out") from exc

    client_any.cancelAccountSummary(req_id)

    positions = await sync_positions()
    return account_snapshot_from_values(
        snapshot_id=uuid.uuid4(),
        as_of=datetime.now(tz=UTC),
        values=values,
        positions=tuple(positions),
    )


async def sync_position_snapshots(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    resolve_instrument_id: Callable[[int], uuid.UUID | None],
) -> list[PositionSnapshot]:
    """Run the IB positions request and return mapped domain position snapshots."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    loop = asyncio.get_running_loop()
    wrapper_any._positions = []
    wrapper_any._positions_expected_generation += 1
    gen = wrapper_any._positions_expected_generation
    wrapper_any._positions_generation = gen
    wrapper_any._positions_done = loop.create_future()

    client_any.reqPositions()

    try:
        raw_positions = await asyncio.wait_for(wrapper_any._positions_done, timeout=timeout)
    except TimeoutError as exc:
        # Invalidate the generation so a late positionEnd is ignored.
        wrapper_any._positions_generation = gen - 1
        client_any.cancelPositions()
        raise BrokerUnavailableError("sync_positions timed out") from exc

    client_any.cancelPositions()
    now = datetime.now(tz=UTC)

    snapshots: list[PositionSnapshot] = []
    for _acct, contract, qty, avg_cost in raw_positions:
        qty_int = int(qty)
        if qty_int <= 0:
            continue

        con_id = contract_con_id(contract) if contract is not None else 0
        instrument_id = resolve_instrument_id(con_id)
        if instrument_id is None:
            log.error(
                "broker_gateway.sync_positions.unmapped_contract",
                con_id=con_id,
                symbol=contract.symbol if contract else None,
                quantity=qty_int,
                detail=(
                    "broker position has no internal instrument mapping; "
                    "add con_id to instrument_contracts to track this position"
                ),
            )
            continue

        if avg_cost <= 0:
            log.error(
                "broker_gateway.sync_positions.missing_price",
                instrument_id=str(instrument_id),
                con_id=con_id,
                symbol=contract.symbol if contract else None,
                quantity=qty_int,
                detail=(
                    "broker returned avg_cost=0 for a live position; "
                    "position skipped to avoid fabricating NAV"
                ),
            )
            continue

        snapshots.append(
            position_snapshot_from_values(
                snapshot_id=uuid.uuid4(),
                instrument_id=instrument_id,
                quantity=qty_int,
                average_cost=Decimal(str(avg_cost)),
                as_of=now,
            )
        )

    return snapshots
