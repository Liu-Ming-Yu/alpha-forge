from __future__ import annotations

from decimal import Decimal

import pytest

from quant_platform.application.operator.requests import RunEngineRequest, SuperviseRequest
from quant_platform.application.results import UseCaseStatus
from quant_platform.bootstrap.engine.loop import EngineLoopConfig, EngineLoopSummary
from quant_platform.bootstrap.operator_adapters import engine as engine_adapter_module
from quant_platform.bootstrap.operator_adapters.engine import EngineAdapters
from quant_platform.bootstrap.operator_adapters.lifecycle import RuntimeAdapters
from quant_platform.config import PlatformSettings, StorageSettings


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
    )


@pytest.mark.asyncio
async def test_run_engine_adapter_delegates_to_shared_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[EngineLoopConfig] = []

    async def _fake_loop(_settings: object, config: EngineLoopConfig) -> EngineLoopSummary:
        captured.append(config)
        return EngineLoopSummary(
            engine_name=config.engine_name,
            mode=config.mode,
            execution_backend=config.execution_backend,
            attempted_cycles=3,
            completed_cycles=3,
            stop_reason="max_cycles_reached",
        )

    monkeypatch.setattr(engine_adapter_module, "run_engine_loop", _fake_loop)

    result = await EngineAdapters(_settings()).run_engine(
        RunEngineRequest(
            mode="paper",
            initial_cash=Decimal("50000"),
            cycles=3,
            contracts_file="contracts.json",
            engine_name="etf_macro_allocator",
            execution_backend="simulated",
        )
    )

    assert result.status is UseCaseStatus.OK
    assert captured == [
        EngineLoopConfig(
            engine_name="etf_macro_allocator",
            mode="paper",
            execution_backend="simulated",
            initial_cash=Decimal("50000"),
            contracts_file="contracts.json",
            interval_seconds=0.0,
            max_cycles=3,
            install_signal_handlers=False,
        )
    ]


@pytest.mark.asyncio
async def test_supervise_adapter_passes_engine_routing_to_supervise_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_platform.bootstrap.engine as bootstrap_engine

    captured: dict[str, object] = {}

    async def _fake_supervise(_settings: object, **kwargs: object) -> EngineLoopSummary:
        captured.update(kwargs)
        return EngineLoopSummary(
            engine_name=str(kwargs["engine_name"]),
            mode=str(kwargs["mode"]),
            execution_backend=str(kwargs["execution_backend"]),
            attempted_cycles=1,
            completed_cycles=1,
            stop_reason="max_cycles_reached",
        )

    monkeypatch.setattr(bootstrap_engine, "supervise_engine", _fake_supervise)

    result = await RuntimeAdapters(_settings()).supervise(
        SuperviseRequest(
            initial_cash=Decimal("50000"),
            interval_seconds=300.0,
            mode="paper",
            max_cycles=1,
            contracts_file="contracts.json",
            engine_name="cross_sectional_equity",
            execution_backend="ib-paper",
        )
    )

    assert result.status is UseCaseStatus.OK
    assert captured == {
        "initial_cash": Decimal("50000"),
        "interval_seconds": 300.0,
        "mode": "paper",
        "max_cycles": 1,
        "contracts_file": "contracts.json",
        "engine_name": "cross_sectional_equity",
        "execution_backend": "ib-paper",
    }
