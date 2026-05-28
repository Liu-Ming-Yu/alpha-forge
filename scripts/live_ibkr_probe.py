#!/usr/bin/env python3
"""Read-only live TWS/IB Gateway probe.

This intentionally requires an instrument contracts file.  A probe without
contracts can connect, but position sync will fail closed for every unknown
broker conId and produce misleading ``positions=0`` output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.application.research.common import _load_instrument_contracts
from quant_platform.config import PlatformSettings
from quant_platform.core.contracts import BrokerHealthStatus
from quant_platform.services.execution_service.gateways.broker_gateway import (
    IBGatewayBrokerGateway,
)

if TYPE_CHECKING:
    from quant_platform.core.domain import PositionSnapshot


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _position_payload(
    instrument_id: uuid.UUID,
    position: PositionSnapshot,
    contracts: dict[uuid.UUID, dict[str, object]],
) -> dict[str, object]:
    spec = contracts.get(instrument_id, {})
    return {
        "instrument_id": instrument_id,
        "symbol": spec.get("symbol"),
        "con_id": spec.get("con_id"),
        "quantity": position.quantity,
        "market_value": position.market_value,
    }


async def _probe(contracts_file: Path) -> dict[str, Any]:
    settings = PlatformSettings()
    contracts = _load_instrument_contracts(str(contracts_file))
    broker = IBGatewayBrokerGateway(
        settings=settings.broker,
        instrument_contracts=contracts,
    )

    await broker.connect()
    try:
        health = await broker.health_check()
        open_orders = await broker.fetch_open_orders()
        account = await broker.sync_account()
        positions = await broker.sync_positions()
        lifecycle_events = await broker.drain_lifecycle_events()
        return {
            "health": health.status.value,
            "health_connected": health.status == BrokerHealthStatus.CONNECTED,
            "latency_ms": health.latency_ms,
            "open_orders": len(open_orders),
            "positions": len(positions),
            "mapped_positions": [
                _position_payload(position.instrument_id, position, contracts)
                for position in positions
            ],
            "lifecycle_events": len(lifecycle_events),
            "nav_available": account.net_asset_value >= 0,
        }
    finally:
        await broker.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contracts-file",
        default=os.environ.get("QP__LIVE_IBKR__CONTRACTS_FILE", ""),
        help=(
            "JSON mapping of internal instrument UUID -> IB contract spec. "
            "Defaults to QP__LIVE_IBKR__CONTRACTS_FILE."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.contracts_file:
        raise SystemExit(
            "pass --contracts-file or set QP__LIVE_IBKR__CONTRACTS_FILE; "
            "live position probes require con_id mappings"
        )
    payload = asyncio.run(_probe(Path(args.contracts_file)))
    print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
