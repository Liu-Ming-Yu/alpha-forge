"""Unit tests for engine session wiring helpers."""

from __future__ import annotations

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.bootstrap.engine import session_wiring
from quant_platform.bootstrap.engine.session_wiring import create_engine_runtime_session
from quant_platform.config import BrokerSettings, ExecutionSettings, PlatformSettings
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.engines.framework.types import EngineConfig, ExecutionBackend, RunMode


def _snapshot() -> AccountSnapshot:
    from datetime import UTC, datetime

    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _contracts() -> dict[uuid.UUID, dict[str, object]]:
    return {
        uuid.uuid4(): {
            "symbol": "AAPL",
            "exchange": "SMART",
            "currency": "USD",
            "sector": "Technology",
            "adv_shares_20d": 10_000_000,
            "last_close": "100",
            "con_id": 265598,
        }
    }


@pytest.mark.asyncio
async def test_create_engine_runtime_session_wires_and_connects_paper_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = SimpleNamespace(broker=SimpleNamespace(connect=AsyncMock()))
    calls: dict[str, object] = {}

    def _create_paper_session(**kwargs: object) -> object:
        calls.update(kwargs)
        return fake_session

    monkeypatch.setattr(session_wiring, "create_paper_session", _create_paper_session)
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())
    signal_model = object()
    portfolio_constructor = object()

    session = await create_engine_runtime_session(
        config=EngineConfig(
            engine_name="equity",
            run_mode=RunMode.PAPER,
            initial_cash=Decimal("100000"),
        ),
        settings=PlatformSettings(_env_file=None),
        strategy_run=strategy_run,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
    )

    assert session is fake_session
    fake_session.broker.connect.assert_awaited_once()
    assert calls["strategy_run_id"] == strategy_run.run_id
    assert calls["signal_model"] is signal_model
    assert calls["portfolio_constructor"] is portfolio_constructor


@pytest.mark.asyncio
async def test_create_engine_runtime_session_requires_live_contracts() -> None:
    with pytest.raises(ValueError, match="LIVE mode requires instrument_contracts"):
        await create_engine_runtime_session(
            config=EngineConfig(
                engine_name="equity",
                run_mode=RunMode.LIVE,
                initial_cash=Decimal("100000"),
            ),
            settings=PlatformSettings(_env_file=None),
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            signal_model=object(),
            portfolio_constructor=object(),
        )


@pytest.mark.asyncio
async def test_create_engine_runtime_session_wires_ib_paper_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = SimpleNamespace(broker=SimpleNamespace(connect=AsyncMock()))
    contracts = _contracts()
    calls: dict[str, object] = {}

    async def _bootstrap_ib_paper_snapshot(**kwargs: object) -> AccountSnapshot:
        calls["bootstrap"] = kwargs
        return _snapshot()

    def _create_ib_paper_session(**kwargs: object) -> object:
        calls["session"] = kwargs
        return fake_session

    monkeypatch.setattr(
        session_wiring,
        "bootstrap_ib_paper_snapshot",
        _bootstrap_ib_paper_snapshot,
    )
    monkeypatch.setattr(
        session_wiring,
        "create_ib_paper_session",
        _create_ib_paper_session,
    )
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())

    session = await create_engine_runtime_session(
        config=EngineConfig(
            engine_name="equity",
            run_mode=RunMode.PAPER,
            execution_backend=ExecutionBackend.IB_PAPER,
            instrument_contracts=contracts,
        ),
        settings=PlatformSettings(
            _env_file=None,
            broker=BrokerSettings(
                host="localhost",
                port=7497,
                paper_trading=True,
                account_id="DU123456",
            ),
        ),
        strategy_run=strategy_run,
        signal_model=object(),
        portfolio_constructor=object(),
    )

    assert session is fake_session
    fake_session.broker.connect.assert_awaited_once()
    assert calls["bootstrap"]["instrument_contracts"] == contracts
    assert calls["session"]["strategy_run_id"] == strategy_run.run_id
    assert calls["session"]["instrument_contracts"] == contracts


@pytest.mark.asyncio
async def test_create_engine_runtime_session_rejects_unsafe_ib_paper_config() -> None:
    with pytest.raises(ValueError, match="PAPER_TRADING=true"):
        await create_engine_runtime_session(
            config=EngineConfig(
                engine_name="equity",
                run_mode=RunMode.PAPER,
                execution_backend=ExecutionBackend.IB_PAPER,
                instrument_contracts=_contracts(),
            ),
            settings=PlatformSettings(
                _env_file=None,
                broker=BrokerSettings(paper_trading=False, port=7497),
                execution=ExecutionSettings(trading_hours_enforced=True),
            ),
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            signal_model=object(),
            portfolio_constructor=object(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("broker_settings", "message"),
    [
        (BrokerSettings(paper_trading=True, port=7496), "paper TWS/Gateway port"),
        (
            BrokerSettings(paper_trading=True, port=7497, account_id="U123456"),
            "DU paper account_id",
        ),
    ],
)
async def test_create_engine_runtime_session_rejects_non_paper_ib_targets(
    broker_settings: BrokerSettings,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await create_engine_runtime_session(
            config=EngineConfig(
                engine_name="equity",
                run_mode=RunMode.PAPER,
                execution_backend=ExecutionBackend.IB_PAPER,
                instrument_contracts=_contracts(),
            ),
            settings=PlatformSettings(_env_file=None, broker=broker_settings),
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            signal_model=object(),
            portfolio_constructor=object(),
        )


@pytest.mark.asyncio
async def test_create_engine_runtime_session_rejects_ib_paper_backend_outside_paper() -> None:
    with pytest.raises(ValueError, match="only valid with run_mode='paper'"):
        await create_engine_runtime_session(
            config=EngineConfig(
                engine_name="equity",
                run_mode=RunMode.SHADOW,
                execution_backend=ExecutionBackend.IB_PAPER,
                instrument_contracts=_contracts(),
            ),
            settings=PlatformSettings(_env_file=None),
            strategy_run=SimpleNamespace(run_id=uuid.uuid4()),
            signal_model=object(),
            portfolio_constructor=object(),
        )
