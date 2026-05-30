"""Read-only TWS/IB-Gateway account snapshot for the operator console.

Lives in the bootstrap (composition) layer because it constructs the concrete
``IBGatewayBrokerGateway`` service adapter — something the view/edge layer is not
allowed to import directly. The view passes already-resolved connection
coordinates (host/port/client id derived from mode) and gets back a plain,
JSON-able snapshot, degrading gracefully when ``ibapi`` is absent or TWS is
unreachable.
"""

from __future__ import annotations

import asyncio
import contextlib
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid

_CONNECT_TIMEOUT = 15.0
_CALL_TIMEOUT = 10.0


def _gateway_type() -> Any:
    try:
        from quant_platform.services.execution_service.gateways.broker_gateway import (
            IBGatewayBrokerGateway,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "ibapi":
            raise RuntimeError("ibapi package is not installed") from exc
        raise
    return IBGatewayBrokerGateway


def _f(value: object) -> float | None:
    # Broker DTOs carry Decimal money values, so accept it alongside the plain
    # numeric/str types (all are convertible to float).
    if not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _position_payload(
    position: object, contracts: dict[uuid.UUID, dict[str, object]]
) -> dict[str, Any]:
    instrument_id = getattr(position, "instrument_id", None)
    spec = contracts.get(instrument_id, {}) if instrument_id is not None else {}
    return {
        "instrument_id": str(instrument_id) if instrument_id is not None else None,
        "symbol": spec.get("symbol"),
        "con_id": spec.get("con_id"),
        "quantity": _f(getattr(position, "quantity", None)),
        "market_value": _f(getattr(position, "market_value", None)),
    }


async def sync_broker_snapshot(
    *,
    host: str,
    port: int,
    client_id: int,
    paper_trading: bool,
    use_gateway: bool,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> dict[str, Any]:
    """Connect read-only, pull health/account/positions/orders, and return them."""
    try:
        gateway_type = _gateway_type()
    except RuntimeError as exc:
        return {"connected": False, "error": str(exc)}

    gateway = gateway_type(
        host=host,
        port=port,
        client_id=client_id,
        paper_trading=paper_trading,
        use_gateway=use_gateway,
        instrument_contracts=instrument_contracts,
    )
    try:
        await asyncio.wait_for(gateway.connect(), timeout=_CONNECT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - surface TWS-unreachable to the operator
        # A timeout cancels connect() mid-handshake, which can leave a half-open
        # socket / reader thread — tear it down before returning.
        await _safe_disconnect(gateway)
        return {"connected": False, "error": f"could not reach TWS: {exc}"}

    try:
        health = await asyncio.wait_for(gateway.health_check(), timeout=_CALL_TIMEOUT)
        account = await asyncio.wait_for(gateway.sync_account(), timeout=_CALL_TIMEOUT)
        positions = await asyncio.wait_for(gateway.sync_positions(), timeout=_CALL_TIMEOUT)
        open_orders = await asyncio.wait_for(gateway.fetch_open_orders(), timeout=_CALL_TIMEOUT)
        status = getattr(getattr(health, "status", None), "value", "unknown")
        return {
            "connected": True,
            "health": {
                "status": status,
                "latency_ms": _f(getattr(health, "latency_ms", None)),
                "last_heartbeat_at": _iso(getattr(health, "last_heartbeat_at", None)),
                "detail": getattr(health, "detail", ""),
            },
            "account": {
                "net_asset_value": _f(getattr(account, "net_asset_value", None)),
                "settled_cash": _f(getattr(account, "settled_cash", None)),
                "unsettled_cash": _f(getattr(account, "unsettled_cash", None)),
                "position_count": len(getattr(account, "positions", []) or []),
            },
            "positions": [_position_payload(p, instrument_contracts) for p in positions],
            "open_orders_count": len(open_orders),
        }
    except Exception as exc:  # noqa: BLE001
        return {"connected": True, "error": f"sync failed after connect: {exc}"}
    finally:
        await _safe_disconnect(gateway)


async def _safe_disconnect(gateway: Any) -> None:
    """Best-effort, bounded teardown of a (possibly half-open) connection."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(gateway.disconnect(), timeout=_CALL_TIMEOUT)
