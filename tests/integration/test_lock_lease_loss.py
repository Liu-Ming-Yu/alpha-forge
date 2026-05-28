"""Chaos test: distributed-lock lease loss during a strategy cycle.

Simulates the scenario where the Redis distributed-lock lease is lost
mid-cycle (e.g. another operator `DEL`'d the key, or Redis failed over and
the TTL expired).  The contract is:

* ``run_strategy_cycle`` must raise ``RuntimeError`` on lease loss.
* The execution policy's kill switch must be activated.
* A ``KillSwitchActivated`` event must be emitted with the lock-lease
  ``activated_by`` tag so operators can triage.

This test runs in-process with a fake lock object to keep CI hermetic;
the real Redis path is exercised in the ``integration_durable`` job.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings, RiskSettings
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.core.events import KillSwitchActivated
from quant_platform.infrastructure.support.clock import FakeClock
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel
from quant_platform.session import create_paper_session, run_strategy_cycle

_UTC = UTC
_NOW = datetime(2026, 2, 3, 10, 0, 0, tzinfo=_UTC)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="chaos_lock_lease",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_NOW,
        started_at=_NOW,
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_id=uuid.uuid4(),
        as_of=_NOW,
        regime_label=RegimeLabel.RISK_ON,
        confidence=1.0,
        detector_version="test",
        supporting_features={},
    )


class _LeaseLostLock:
    """Fake lock that reports lease_lost=True after release."""

    def __init__(self) -> None:
        self.lease_lost = False
        self._entered = False

    async def acquire(self) -> bool:
        self._entered = True
        return True

    async def release(self) -> None:
        # Simulate lease loss being detected during the cycle.
        self.lease_lost = True

    async def __aenter__(self) -> _LeaseLostLock:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()


@pytest.mark.asyncio
async def test_lock_lease_loss_activates_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock(_NOW)
    instrument = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.20"),
            max_sector_weight=Decimal("0.50"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.30"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
    )
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("100000"),
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )
    session.broker.set_market_price(instrument, Decimal("100"))  # type: ignore[attr-defined]
    await session.broker.connect()

    # Patch the lock factory used by run_strategy_cycle so the simulated
    # cycle thinks the distributed lock lost its lease mid-flight.  We use
    # ``monkeypatch`` instead of direct attribute assignment so pytest
    # restores the original factory on teardown — without it, every
    # subsequent test that calls ``run_strategy_cycle`` in the same
    # process would inherit the fake lease-losing lock and fail.
    import quant_platform.engines.session.public_api as session_module

    def _fake_factory(*_args: object, **_kwargs: object) -> _LeaseLostLock:
        return _LeaseLostLock()

    monkeypatch.setattr(session_module, "create_distributed_lock", _fake_factory)

    with pytest.raises(RuntimeError, match="lock lease was lost"):
        await run_strategy_cycle(
            session=session,
            feature_data={instrument: {"momentum": 0.9}},
            strategy_run=_strategy_run(),
            market_prices={instrument: Decimal("100")},
            regime=_regime(),
        )

    kill_events = [e for e in session.event_bus.history if isinstance(e, KillSwitchActivated)]
    assert kill_events, "KillSwitchActivated must be emitted on lease loss"
    assert any(e.activated_by == "distributed_lock" for e in kill_events), (
        "kill-switch event must tag the lock-lease origin"
    )
    assert session.execution_policy.kill_switch_active


class _PreSubmitLeaseLossLock:
    """Fake lock that reports ``lease_lost`` *before* order submission.

    Used to assert that the strategy cycle aborts with the broker pristine
    when the lease is lost between order approval and submit_cycle_orders.
    """

    def __init__(self) -> None:
        self.lease_lost = True  # already lost before any release
        self._entered = False

    async def acquire(self) -> bool:
        self._entered = True
        return True

    async def release(self) -> None:
        return

    async def __aenter__(self) -> _PreSubmitLeaseLossLock:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()


@pytest.mark.asyncio
async def test_lease_lost_before_submit_skips_broker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the lock-lease is lost between order approval and submission, the
    cycle must abort *before* dispatching to the broker.  Regression for
    audit finding M-7 (verifying the pre-submit lease check is in the
    correct position).
    """
    clock = FakeClock(_NOW)
    instrument = uuid.uuid4()
    settings = PlatformSettings(
        _env_file=None,
        risk=RiskSettings(
            max_single_name_weight=Decimal("0.20"),
            max_sector_weight=Decimal("0.50"),
            max_gross_exposure=Decimal("0.95"),
            max_daily_turnover=Decimal("0.30"),
            min_cash_buffer=Decimal("0.05"),
            max_drawdown_halt=Decimal("-0.20"),
        ),
    )
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("100000"),
        clock=clock,
        signal_model=LinearWeightSignalModel({"momentum": 1.0}),
    )
    session.broker.set_market_price(instrument, Decimal("100"))  # type: ignore[attr-defined]
    await session.broker.connect()

    # Track whether the broker.place_order path is reached.
    place_order_calls: list[object] = []
    real_place_order = session.broker.place_order

    async def _spy_place_order(*args: object, **kwargs: object) -> object:
        place_order_calls.append(args)
        return await real_place_order(*args, **kwargs)  # type: ignore[misc]

    session.broker.place_order = _spy_place_order  # type: ignore[assignment]

    import quant_platform.engines.session.public_api as session_module

    monkeypatch.setattr(
        session_module,
        "create_distributed_lock",
        lambda *a, **k: _PreSubmitLeaseLossLock(),
    )

    with pytest.raises(RuntimeError, match="lease"):
        await run_strategy_cycle(
            session=session,
            feature_data={instrument: {"momentum": 0.9}},
            strategy_run=_strategy_run(),
            market_prices={instrument: Decimal("100")},
            regime=_regime(),
        )

    assert place_order_calls == [], (
        "broker.place_order must NOT be called when the lock-lease is lost"
        " before submit; pre-submit lease check is the safety gate."
    )
