"""ADR-014: supervise routes through the account-orchestrator runner under V2."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from quant_platform.bootstrap import engine as engine_mod
from quant_platform.bootstrap.engine.loop_types import EngineLoopSummary
from quant_platform.bootstrap.engine.orchestrator_runner import AccountOrchestratorLoopRunner
from quant_platform.config import PlatformSettings, V2Settings

if TYPE_CHECKING:
    import pytest as _pytest


def _summary(config: Any) -> EngineLoopSummary:
    return EngineLoopSummary(
        engine_name=config.engine_name,
        mode=config.mode,
        execution_backend=config.execution_backend,
    )


async def _run(monkeypatch: _pytest.MonkeyPatch, *, v2_enabled: bool) -> Any:
    captured: dict[str, Any] = {}

    async def fake_loop(settings: Any, config: Any, *, runner_factory: Any = None, **_: Any) -> Any:
        captured["runner_factory"] = runner_factory
        return _summary(config)

    monkeypatch.setattr(engine_mod, "run_engine_loop", fake_loop)
    settings = PlatformSettings(
        _env_file=None,
        v2=V2Settings(enabled=v2_enabled, account_orchestrator_enabled=v2_enabled),
    )
    await engine_mod.supervise_engine(
        settings,
        initial_cash=Decimal("50000"),
        interval_seconds=0.0,
        mode="paper",
        max_cycles=1,
        engine_name="cross_sectional_equity",
        execution_backend="simulated",
    )
    return captured


@pytest.mark.asyncio
async def test_supervise_uses_orchestrator_runner_when_v2_enabled(
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    captured = await _run(monkeypatch, v2_enabled=True)
    factory = captured["runner_factory"]
    assert factory is not None
    runner = factory()
    assert isinstance(runner, AccountOrchestratorLoopRunner)


@pytest.mark.asyncio
async def test_supervise_falls_back_to_v1_when_v2_disabled(
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    captured = await _run(monkeypatch, v2_enabled=False)
    # No runner_factory ⇒ run_engine_loop builds the default V1 EngineRunner.
    assert captured["runner_factory"] is None
