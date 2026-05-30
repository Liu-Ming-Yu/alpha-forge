"""Unit tests for the read-only broker data-sync teardown semantics.

Focus: a connect failure (e.g. a half-open TWS handshake that times out) must
still tear the connection down, otherwise repeated syncs leak sockets/threads.
"""

from __future__ import annotations

import asyncio
from typing import Any

from quant_platform.bootstrap.broker import data_sync


def test_connect_failure_still_disconnects(monkeypatch: Any) -> None:
    events: list[str] = []

    class _FakeGateway:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def connect(self) -> None:
            raise TimeoutError("handshake stalled")

        async def disconnect(self) -> None:
            events.append("disconnect")

    monkeypatch.setattr(data_sync, "_gateway_type", lambda: _FakeGateway)

    result = asyncio.run(
        data_sync.sync_broker_snapshot(
            host="127.0.0.1",
            port=7497,
            client_id=99,
            paper_trading=True,
            use_gateway=False,
            instrument_contracts={},
        )
    )

    assert result["connected"] is False
    assert "could not reach TWS" in result["error"]
    # The half-open connection was cleaned up despite connect() failing.
    assert events == ["disconnect"]


def test_successful_sync_disconnects_in_finally(monkeypatch: Any) -> None:
    events: list[str] = []

    class _Health:
        status = type("S", (), {"value": "connected"})()
        latency_ms = 1.5
        last_heartbeat_at = None
        detail = "ok"

    class _Account:
        net_asset_value = 1000.0
        settled_cash = 900.0
        unsettled_cash = 0.0
        positions: list[Any] = []

    class _FakeGateway:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def connect(self) -> None:
            events.append("connect")

        async def health_check(self) -> _Health:
            return _Health()

        async def sync_account(self) -> _Account:
            return _Account()

        async def sync_positions(self) -> list[Any]:
            return []

        async def fetch_open_orders(self) -> list[Any]:
            return []

        async def disconnect(self) -> None:
            events.append("disconnect")

    monkeypatch.setattr(data_sync, "_gateway_type", lambda: _FakeGateway)

    result = asyncio.run(
        data_sync.sync_broker_snapshot(
            host="127.0.0.1",
            port=7497,
            client_id=99,
            paper_trading=True,
            use_gateway=False,
            instrument_contracts={},
        )
    )

    assert result["connected"] is True
    assert result["account"]["net_asset_value"] == 1000.0
    # Connection always torn down after a successful pull.
    assert events == ["connect", "disconnect"]
