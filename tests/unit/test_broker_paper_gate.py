from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.application.operator.cli_inputs import load_instrument_contracts
from quant_platform.bootstrap.broker import (
    broker_gate_settings,
    classify_broker_probe_failure,
    ib_gateway_smoke,
    ib_paper_lifecycle,
    paper_lifecycle_limit_price,
)
from quant_platform.config import BrokerSettings, ExecutionSettings, PlatformSettings
from quant_platform.core.contracts import BrokerAck
from quant_platform.core.domain.orders import BrokerOrder, OrderStatus


def _contracts_file(tmp_path, *, con_id: object = 265598) -> tuple[uuid.UUID, str]:  # type: ignore[no-untyped-def]
    instrument_id = uuid.uuid4()
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps(
            {
                str(instrument_id): {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "con_id": con_id,
                    "last_close": "190",
                }
            }
        ),
        encoding="utf-8",
    )
    return instrument_id, str(path)


def test_broker_gate_settings_respect_configured_host_and_port() -> None:
    settings = PlatformSettings(_env_file=None)

    broker = broker_gate_settings(settings)

    assert broker.host == "127.0.0.1"
    assert broker.port == 7497


def test_broker_probe_failure_classification() -> None:
    assert (
        classify_broker_probe_failure(
            RuntimeError("IB Gateway socket connection failed: connection refused")
        )
        == "socket_failure"
    )
    assert (
        classify_broker_probe_failure(
            RuntimeError("IB Gateway did not send nextValidId within 60s")
        )
        == "handshake_timeout"
    )
    assert (
        classify_broker_probe_failure(RuntimeError("Trusted IPs rejected WSL client"))
        == "auth_or_trusted_ip"
    )


@pytest.mark.asyncio
async def test_paper_lifecycle_refuses_live_account(tmp_path) -> None:
    instrument_id, path = _contracts_file(tmp_path)
    settings = PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(paper_trading=False),
        execution=ExecutionSettings(trading_hours_enforced=True),
    )

    with pytest.raises(ValueError, match="paper_trading=true"):
        await _ib_paper_lifecycle(
            settings,
            contracts_file=path,
            instrument_id=instrument_id,
            max_notional_usd=Decimal("100"),
        )


@pytest.mark.asyncio
async def test_paper_lifecycle_refuses_excessive_notional(tmp_path) -> None:
    instrument_id, path = _contracts_file(tmp_path)

    with pytest.raises(ValueError, match="max-notional-usd"):
        await _ib_paper_lifecycle(
            PlatformSettings(_env_file=None),
            contracts_file=path,
            instrument_id=instrument_id,
            max_notional_usd=Decimal("101"),
        )


@pytest.mark.asyncio
async def test_paper_lifecycle_refuses_missing_con_id(tmp_path) -> None:
    instrument_id, path = _contracts_file(tmp_path, con_id=None)

    with pytest.raises(ValueError, match="positive integer con_id"):
        await _ib_paper_lifecycle(
            PlatformSettings(_env_file=None),
            contracts_file=path,
            instrument_id=instrument_id,
            max_notional_usd=Decimal("100"),
        )


def test_paper_lifecycle_limit_is_non_marketable_from_last_close() -> None:
    limit = paper_lifecycle_limit_price({"last_close": "190"}, Decimal("100"))

    assert limit == Decimal("95.00")


@pytest.mark.asyncio
async def test_ib_gateway_smoke_success_reports_all_read_only_steps(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument_id, path = _contracts_file(tmp_path)
    calls: list[str] = []

    class _FakeGateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("init")

        async def connect(self) -> None:
            calls.append("connect")

        async def health_check(self) -> object:
            calls.append("health")
            return SimpleNamespace(
                status=SimpleNamespace(value="connected"),
                latency_ms=3.5,
                last_heartbeat_at=datetime.now(tz=UTC),
                detail="ok",
            )

        async def sync_account(self) -> object:
            calls.append("account")
            return SimpleNamespace(
                as_of=datetime.now(tz=UTC),
                net_asset_value=Decimal("100000"),
                settled_cash=Decimal("100000"),
            )

        async def sync_positions(self) -> list[object]:
            calls.append("positions")
            return []

        async def fetch_open_orders(self) -> list[object]:
            calls.append("open_orders")
            return []

        async def disconnect(self) -> None:
            calls.append("disconnect")

    import quant_platform.services.execution_service.gateways.broker_gateway as broker_mod

    monkeypatch.setattr(broker_mod, "IBGatewayBrokerGateway", _FakeGateway)

    report = await _ib_gateway_smoke(
        PlatformSettings(_env_file=None),
        contracts_file=path,
    )

    assert instrument_id
    assert report["passed"] is True
    assert report["account_status"] == "ok"
    assert report["positions_status"] == "ok"
    assert report["open_orders_status"] == "ok"
    assert calls == [
        "init",
        "connect",
        "health",
        "account",
        "positions",
        "open_orders",
        "disconnect",
    ]


@pytest.mark.asyncio
async def test_ib_paper_lifecycle_success_submit_cancel_reconcile(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument_id, path = _contracts_file(tmp_path)
    order_id_seen: uuid.UUID | None = None
    broker_order_id = "101"

    class _FakeGateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._fetch_count = 0

        async def connect(self) -> None:
            pass

        async def place_order(self, intent: object) -> BrokerAck:
            nonlocal order_id_seen
            order_id_seen = intent.order_id  # type: ignore[attr-defined]
            return BrokerAck(
                order_id=order_id_seen,
                broker_order_id=broker_order_id,
                acknowledged_at=datetime.now(tz=UTC),
            )

        async def fetch_open_orders(self) -> list[BrokerOrder]:
            self._fetch_count += 1
            if self._fetch_count == 1:
                assert order_id_seen is not None
                return [
                    BrokerOrder(
                        order_id=order_id_seen,
                        broker_order_id=broker_order_id,
                        status=OrderStatus.SUBMITTED,
                        last_updated_at=datetime.now(tz=UTC),
                    )
                ]
            return []

        async def cancel_order(self, requested_broker_order_id: str) -> None:
            assert requested_broker_order_id == broker_order_id

        async def disconnect(self) -> None:
            pass

    import quant_platform.services.execution_service.gateways.broker_gateway as broker_mod

    monkeypatch.setattr(broker_mod, "IBGatewayBrokerGateway", _FakeGateway)

    report = await _ib_paper_lifecycle(
        PlatformSettings(_env_file=None),
        contracts_file=path,
        instrument_id=instrument_id,
        max_notional_usd=Decimal("100"),
    )

    assert report["passed"] is True
    assert report["broker_order_id"] == broker_order_id
    assert report["ack_status"] == "ok"
    assert report["cancel_status"] == "ok"
    assert report["stale_open_order_count"] == 0


async def _ib_gateway_smoke(
    settings: PlatformSettings,
    *,
    contracts_file: str,
) -> dict[str, object]:
    return await ib_gateway_smoke(settings, load_instrument_contracts(contracts_file))


async def _ib_paper_lifecycle(
    settings: PlatformSettings,
    *,
    contracts_file: str,
    instrument_id: uuid.UUID,
    max_notional_usd: Decimal,
) -> dict[str, object]:
    return await ib_paper_lifecycle(
        settings,
        contracts=load_instrument_contracts(contracts_file),
        instrument_id=instrument_id,
        max_notional_usd=max_notional_usd,
        max_allowed_notional=Decimal("100"),
    )
