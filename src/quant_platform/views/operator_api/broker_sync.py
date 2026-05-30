"""Mode-aware, read-only TWS/IB-Gateway data sync for the operator console.

``connection_info`` reports the resolved connection target for a mode (paper →
paper port, live → live port) without opening a socket. ``sync_broker`` resolves
those coordinates, loads the instrument contracts, and delegates the actual
connect-and-pull to the bootstrap layer (which owns the broker adapter), then
merges the snapshot back with the connection metadata.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings


def _ibapi_available() -> bool:
    try:
        import ibapi  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _mask_account(account_id: str) -> str:
    aid = (account_id or "").strip()
    if not aid:
        return ""
    return aid[:2] + "•" * max(0, len(aid) - 2)


def _contracts_path() -> str:
    return os.environ.get("QP__LIVE_IBKR__CONTRACTS_FILE", "").strip()


# Cache the parsed contracts by (path, mtime) so the polled connection/sync
# endpoints don't re-read + parse the (~330-row) universe file every call.
_ContractSig = tuple[str, int]
_contracts_cache: tuple[_ContractSig, dict[uuid.UUID, dict[str, object]]] | None = None
_contracts_lock = threading.Lock()


def _load_contracts() -> dict[uuid.UUID, dict[str, object]]:
    global _contracts_cache
    path = _contracts_path()
    if not path or not os.path.isfile(path):
        return {}
    try:
        sig: _ContractSig = (path, os.stat(path).st_mtime_ns)
    except OSError:
        return {}
    with _contracts_lock:
        if _contracts_cache is not None and _contracts_cache[0] == sig:
            return _contracts_cache[1]
    try:
        from quant_platform.application.operator.cli_inputs import load_instrument_contracts

        contracts = load_instrument_contracts(path)
    except Exception:  # noqa: BLE001 - missing/invalid contracts must not break the sync
        return {}
    with _contracts_lock:
        _contracts_cache = (sig, contracts)
    return contracts


def connection_info(settings: PlatformSettings, mode: str) -> dict[str, Any]:
    """Resolved connection target for ``mode`` — no socket is opened."""
    broker = settings.broker
    return {
        "mode": mode,
        "host": broker.resolved_host(),
        "port": broker.resolved_port(mode),
        "use_gateway": bool(broker.use_gateway),
        "broker_kind": "IB Gateway" if broker.use_gateway else "TWS",
        "client_path": broker.primary_broker_path,
        "paper_trading": broker.resolved_paper_trading(mode),
        "sync_client_id": broker.sync_client_id(),
        "trading_client_id": broker.client_id,
        "account_id_masked": _mask_account(broker.account_id),
        "ibapi_available": _ibapi_available(),
        "contracts_file": _contracts_path() or None,
        "contracts_count": len(_load_contracts()),
        "ports": {
            "paper": broker.resolved_port("paper"),
            "live": broker.resolved_port("live"),
        },
    }


# Serialize syncs: every sync uses the same dedicated read-only client id, and
# IB rejects a second connection on an id already in use — so overlapping calls
# (e.g. the 30 s poll racing a manual refresh) must run one at a time.
_sync_lock = asyncio.Lock()


async def sync_broker(settings: PlatformSettings, mode: str) -> dict[str, Any]:
    """Connect at the mode-resolved TWS port and pull all account data."""
    info = connection_info(settings, mode)
    if not info["ibapi_available"]:
        info["synced_at"] = datetime.now(UTC).isoformat()
        return {**info, "connected": False, "error": "ibapi package is not installed"}

    from quant_platform.bootstrap.broker.data_sync import sync_broker_snapshot

    contracts = _load_contracts()

    def _blocking() -> dict[str, Any]:
        # Run the whole connect-and-pull on its own loop in a worker thread so
        # a slow/half-open TWS handshake cannot stall the API event loop.
        return asyncio.run(
            sync_broker_snapshot(
                host=info["host"],
                port=info["port"],
                client_id=info["sync_client_id"],
                paper_trading=info["paper_trading"],
                use_gateway=settings.broker.use_gateway,
                instrument_contracts=contracts,
            )
        )

    async with _sync_lock:
        info["synced_at"] = datetime.now(UTC).isoformat()
        snapshot = await asyncio.to_thread(_blocking)
    return {**info, **snapshot}
