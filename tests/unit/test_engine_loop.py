from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quant_platform.application.runtime.state import CycleResult
from quant_platform.bootstrap.engine.loop import EngineLoopConfig, run_engine_loop
from quant_platform.config import BrokerSettings, PlatformSettings, StorageSettings
from quant_platform.core.contracts import BrokerHealthStatus
from quant_platform.engines.framework.types import EngineRunResult, RunMode

_NOW = datetime(2026, 1, 5, 14, 0, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return _NOW

    def today(self) -> date:
        return _NOW.date()


class _FakeRunner:
    def __init__(
        self, *, fail_cycles: set[int] | None = None, session: object | None = None
    ) -> None:
        self.fail_cycles = fail_cycles or set()
        self._session = session
        self.initialized = 0
        self.shutdowns = 0
        self.run_calls = 0

    async def initialize(self) -> None:
        self.initialized += 1

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
    ) -> CycleResult:
        self.run_calls += 1
        if self.run_calls in self.fail_cycles:
            raise RuntimeError(f"cycle {self.run_calls} failed")
        return CycleResult(
            signals=[], target=None, approved=[], rejected=[], submitted_ids=[], fills=[]
        )

    async def shutdown(self) -> EngineRunResult:
        self.shutdowns += 1
        return EngineRunResult(
            run_id=uuid.uuid4(),
            engine_name="fake",
            run_mode=RunMode.PAPER,
            cycles_completed=self.run_calls,
        )


class _FactoryAwareRunner:
    def __init__(self) -> None:
        self._session = None
        self.initialized_with: tuple[object, object] | None = None
        self.run_calls = 0
        self.shutdowns = 0

    async def initialize(self, *, session_factory: object, scheduler_factory: object) -> None:
        self.initialized_with = (session_factory, scheduler_factory)

    async def run_cycle(
        self,
        feature_data: dict[uuid.UUID, dict[str, float]],
    ) -> CycleResult:
        self.run_calls += 1
        return CycleResult(
            signals=[], target=None, approved=[], rejected=[], submitted_ids=[], fills=[]
        )

    async def shutdown(self) -> EngineRunResult:
        self.shutdowns += 1
        return EngineRunResult(
            run_id=uuid.uuid4(),
            engine_name="factory-aware",
            run_mode=RunMode.PAPER,
            cycles_completed=self.run_calls,
        )


class _Policy:
    def __init__(self, active: bool) -> None:
        self.kill_switch_active = active
        self.reason = ""

    def hydrate_kill_switch(self, *, active: bool, reason: str | None = None) -> None:
        self.kill_switch_active = active
        self.reason = reason or ""


class _KillSwitchStore:
    def __init__(self, active: bool) -> None:
        self.active = active

    async def get(self) -> SimpleNamespace:
        return SimpleNamespace(active=self.active, reason="operator review")


class _AccountBroker:
    async def health_check(self) -> SimpleNamespace:
        return SimpleNamespace(status=BrokerHealthStatus.CONNECTED)

    async def fetch_open_orders(self) -> list[object]:
        return []

    async def sync_positions(self) -> list[object]:
        return []

    async def sync_account(self) -> SimpleNamespace:
        return SimpleNamespace(source="broker", settled_cash=Decimal("10000"))


async def _no_sleep(_shutdown: object, _interval: float) -> None:
    return None


def _settings(**overrides: object) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        **overrides,
    )


@pytest.mark.asyncio
async def test_engine_loop_initializes_once_isolates_cycle_errors_and_shutdowns() -> None:
    runner = _FakeRunner(fail_cycles={2})

    summary = await run_engine_loop(
        _settings(),
        EngineLoopConfig(
            engine_name="cross_sectional_equity",
            mode="paper",
            execution_backend="simulated",
            initial_cash=Decimal("10000"),
            contracts_file=None,
            max_cycles=3,
            install_signal_handlers=False,
        ),
        runner_factory=lambda: runner,
        sleep_fn=_no_sleep,
    )

    assert runner.initialized == 1
    assert runner.run_calls == 3
    assert runner.shutdowns == 1
    assert summary.attempted_cycles == 3
    assert summary.completed_cycles == 2
    assert [error.cycle for error in summary.errors] == [2]
    assert summary.stop_reason == "max_cycles_reached"


@pytest.mark.asyncio
async def test_engine_loop_default_runner_injects_engine_factories(monkeypatch) -> None:
    from quant_platform.bootstrap.engine import multi as multi_mod

    runner = _FactoryAwareRunner()

    monkeypatch.setattr(multi_mod, "create_single_engine_runner", lambda **_kwargs: runner)

    summary = await run_engine_loop(
        _settings(),
        EngineLoopConfig(
            engine_name="cross_sectional_equity",
            mode="paper",
            execution_backend="simulated",
            initial_cash=Decimal("10000"),
            contracts_file=None,
            max_cycles=1,
            install_signal_handlers=False,
        ),
        sleep_fn=_no_sleep,
    )

    assert runner.initialized_with is not None
    assert runner.run_calls == 1
    assert runner.shutdowns == 1
    assert summary.completed_cycles == 1


@pytest.mark.asyncio
async def test_engine_loop_stops_after_shutdown_requested_during_sleep() -> None:
    runner = _FakeRunner()

    async def _stop(shutdown: object, _interval: float) -> None:
        shutdown.set()

    summary = await run_engine_loop(
        _settings(),
        EngineLoopConfig(
            engine_name="cross_sectional_equity",
            mode="paper",
            execution_backend="simulated",
            initial_cash=Decimal("10000"),
            contracts_file=None,
            interval_seconds=1.0,
            max_cycles=None,
            install_signal_handlers=False,
        ),
        runner_factory=lambda: runner,
        sleep_fn=_stop,
    )

    assert runner.run_calls == 1
    assert runner.shutdowns == 1
    assert summary.stop_reason == "shutdown_requested"


@pytest.mark.asyncio
async def test_engine_loop_blocks_while_kill_switch_active_then_resumes_after_clear() -> None:
    store = _KillSwitchStore(active=True)
    policy = _Policy(active=True)
    session = SimpleNamespace(
        execution_policy=policy,
        kill_switch_store=store,
        account_broker=_AccountBroker(),
        position_repo=SimpleNamespace(get_latest_snapshot=lambda: _latest_snapshot()),
        clock=_Clock(),
        settings=_settings(),
        cash_engine=SimpleNamespace(settled_cash=Decimal("10000")),
        event_bus=SimpleNamespace(),
    )
    runner = _FakeRunner(session=session)

    async def _clear_after_blocked_tick(_shutdown: object, _interval: float) -> None:
        store.active = False

    summary = await run_engine_loop(
        _settings(),
        EngineLoopConfig(
            engine_name="cross_sectional_equity",
            mode="paper",
            execution_backend="simulated",
            initial_cash=Decimal("10000"),
            contracts_file=None,
            interval_seconds=1.0,
            max_cycles=2,
            install_signal_handlers=False,
        ),
        runner_factory=lambda: runner,
        sleep_fn=_clear_after_blocked_tick,
    )

    assert summary.attempted_cycles == 2
    assert summary.blocked_cycles == 1
    assert summary.completed_cycles == 1
    assert summary.last_recovery_assessment is not None
    assert summary.last_recovery_assessment.ready_for_operator_clear is True
    assert policy.kill_switch_active is False


@pytest.mark.asyncio
async def test_engine_loop_validates_ib_paper_contracts_before_start(tmp_path) -> None:
    contracts = tmp_path / "contracts.json"
    instrument_id = uuid.uuid4()
    contracts.write_text(
        f'{{"{instrument_id}": {{"symbol": "AAPL", "exchange": "SMART", "currency": "USD"}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive con_id"):
        await run_engine_loop(
            _settings(
                broker=BrokerSettings(
                    paper_trading=True,
                    port=7497,
                    account_id="DU123456",
                )
            ),
            EngineLoopConfig(
                engine_name="cross_sectional_equity",
                mode="paper",
                execution_backend="ib-paper",
                initial_cash=Decimal("10000"),
                contracts_file=str(contracts),
                max_cycles=1,
                install_signal_handlers=False,
            ),
            runner_factory=lambda: _FakeRunner(),
            sleep_fn=_no_sleep,
        )


@pytest.mark.asyncio
async def test_engine_loop_rejects_ib_paper_outside_paper_mode() -> None:
    with pytest.raises(ValueError, match="only valid with --mode paper"):
        await run_engine_loop(
            _settings(),
            EngineLoopConfig(
                engine_name="cross_sectional_equity",
                mode="shadow",
                execution_backend="ib-paper",
                initial_cash=Decimal("10000"),
                contracts_file=None,
                max_cycles=1,
                install_signal_handlers=False,
            ),
            runner_factory=lambda: _FakeRunner(),
            sleep_fn=_no_sleep,
        )


async def _latest_snapshot() -> SimpleNamespace:
    return SimpleNamespace(positions=())
