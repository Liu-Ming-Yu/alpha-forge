"""IB-paper session wiring with a fake broker adapter.

This keeps the brokered paper path hermetic while proving the session factory
uses the IB-style order-routing adapter instead of the simulated broker.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from quant_platform.bootstrap.broker import live_broker_wiring
from quant_platform.config import BrokerSettings, PlatformSettings
from quant_platform.core.contracts import (
    BrokerAck,
    BrokerCapabilities,
    BrokerHealth,
    BrokerHealthStatus,
)
from quant_platform.core.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.session import create_ib_paper_session

_NOW = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
_INSTRUMENT_ID = uuid.uuid4()


def _snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_NOW,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _contracts() -> dict[uuid.UUID, dict[str, object]]:
    return {
        _INSTRUMENT_ID: {
            "symbol": "AAPL",
            "exchange": "SMART",
            "currency": "USD",
            "sector": "Technology",
            "adv_shares_20d": 10_000_000,
            "last_close": "100",
            "con_id": 265598,
        }
    }


class _FakeIBPaperGateway:
    instances: list[_FakeIBPaperGateway] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.connected = False
        self.placed_orders: list[OrderIntent] = []
        self.execution_policy = None
        self._capabilities = BrokerCapabilities(
            provider="fake_ib_paper",
            supports_order_routing=True,
            supports_order_cancellation=True,
            supports_lifecycle_feed=True,
        )
        self.instances.append(self)

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self._capabilities

    def validate_instrument_mappings(self) -> list[str]:
        return []

    def set_execution_policy(self, policy: object) -> None:
        self.execution_policy = policy

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def health_check(self) -> BrokerHealth:
        return BrokerHealth(
            status=BrokerHealthStatus.CONNECTED,
            latency_ms=1,
            last_heartbeat_at=_NOW,
        )

    async def sync_account(self) -> AccountSnapshot:
        return _snapshot()

    async def sync_positions(self) -> list[object]:
        return []

    async def fetch_open_orders(self) -> list[object]:
        return []

    async def place_order(self, order: OrderIntent) -> BrokerAck:
        self.placed_orders.append(order)
        return BrokerAck(
            order_id=order.order_id,
            broker_order_id=f"paper-{len(self.placed_orders)}",
            acknowledged_at=_NOW,
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        return None

    async def drain_lifecycle_events(self) -> list[object]:
        return []


@pytest.mark.asyncio
async def test_ib_paper_session_uses_brokered_gateway_for_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeIBPaperGateway.instances.clear()
    monkeypatch.setattr(
        live_broker_wiring,
        "_load_ib_gateway_type",
        lambda: _FakeIBPaperGateway,
    )

    settings = PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(
            host="localhost",
            port=7497,
            paper_trading=True,
            account_id="DU123456",
        ),
    )
    session = create_ib_paper_session(
        settings=settings,
        initial_snapshot=_snapshot(),
        strategy_run_id=uuid.uuid4(),
        instrument_contracts=_contracts(),
    )

    gateway = _FakeIBPaperGateway.instances[-1]
    assert session.broker is gateway
    assert session.account_broker is gateway
    assert session.trading_broker is gateway
    assert gateway.execution_policy is session.execution_policy
    assert len(session.contract_master.list_active()) == 1

    intent = OrderIntent(
        order_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        portfolio_target_id=uuid.uuid4(),
        instrument_id=_INSTRUMENT_ID,
        side=OrderSide.SELL,
        quantity=1,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
        limit_price=Decimal("100"),
    )

    submitted = await session.submit_ctrl.submit([intent])

    assert submitted == [intent.order_id]
    assert gateway.placed_orders == [intent]
