"""Unit tests for mode→TWS port resolution and the broker connection endpoint."""

from __future__ import annotations

import pytest

from quant_platform.config import ApiSettings, BrokerSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api.broker_sync import connection_info


def test_resolved_port_tws_default() -> None:
    broker = BrokerSettings()
    assert broker.resolved_port("paper") == 7497
    assert broker.resolved_port("shadow") == 7497
    assert broker.resolved_port("live") == 7496
    assert broker.resolved_port("LIVE") == 7496  # case-insensitive


def test_resolved_port_gateway_flag() -> None:
    broker = BrokerSettings(use_gateway=True)
    assert broker.resolved_port("paper") == 4002
    assert broker.resolved_port("live") == 4001


def test_resolved_port_infers_gateway_from_pinned_port() -> None:
    broker = BrokerSettings(port=4002)
    assert broker.resolved_port("paper") == 4002
    assert broker.resolved_port("live") == 4001


def test_resolved_paper_trading_and_sync_client_id() -> None:
    broker = BrokerSettings(client_id=5)
    assert broker.resolved_paper_trading("paper") is True
    assert broker.resolved_paper_trading("shadow") is True
    assert broker.resolved_paper_trading("live") is False
    assert broker.sync_client_id() == 95  # client_id + 90
    assert BrokerSettings(client_id=5, read_only_client_id=42).sync_client_id() == 42


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        api=ApiSettings(operator_api_key="test-key"),
        broker=BrokerSettings(paper_trading=True, account_id="DU1234567"),
    )


def test_connection_info_is_mode_aware() -> None:
    settings = _settings()
    paper = connection_info(settings, "paper")
    live = connection_info(settings, "live")
    assert paper["port"] == 7497
    assert live["port"] == 7496
    assert paper["ports"] == {"paper": 7497, "live": 7496}
    assert paper["broker_kind"] == "TWS"
    assert paper["account_id_masked"].startswith("DU")
    assert "1234567" not in paper["account_id_masked"]  # account masked


def test_connection_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from quant_platform.views.operator_api.app import create_app

    client = TestClient(create_app(settings=_settings()))
    headers = {"X-API-Key": "test-key"}
    paper = client.get("/v1/broker/connection?mode=paper", headers=headers)
    assert paper.status_code == 200
    assert paper.json()["port"] == 7497
    live = client.get("/v1/broker/connection?mode=live", headers=headers)
    assert live.json()["port"] == 7496
    assert client.get("/v1/broker/connection").status_code == 401  # auth required
