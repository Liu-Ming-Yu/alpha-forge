"""Unit tests for shadow target-cycle orchestration."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.engines.shadow.cycle import (
    run_shadow_target_cycle,
    save_shadow_nav_snapshot,
)

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


def _account() -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=_AS_OF,
        settled_cash=Decimal("100000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("100000"),
        net_asset_value=Decimal("100000"),
        positions=(),
    )


def _session(*, target: object | None) -> SimpleNamespace:
    regime = SimpleNamespace(regime_label=SimpleNamespace(value="risk_on"))
    return SimpleNamespace(
        clock=SimpleNamespace(now=lambda: _AS_OF),
        account_broker=SimpleNamespace(sync_account=AsyncMock(return_value=_account())),
        signal_ctrl=SimpleNamespace(generate=AsyncMock(return_value=["signal"])),
        regime_detector=SimpleNamespace(detect=AsyncMock(return_value=regime)),
        portfolio_ctrl=SimpleNamespace(build=AsyncMock(return_value=target)),
        order_planner=SimpleNamespace(plan=MagicMock(return_value=["intent"])),
        approve_ctrl=SimpleNamespace(approve=AsyncMock(return_value=(["approved"], ["rejected"]))),
        performance_repo=SimpleNamespace(save_nav_snapshot=AsyncMock()),
        risk_limits=object(),
    )


def _strategy_run() -> SimpleNamespace:
    return SimpleNamespace(run_id=uuid.uuid4(), strategy_name="equity")


@pytest.mark.asyncio
async def test_shadow_target_cycle_builds_and_approves_without_submission() -> None:
    instrument_id = uuid.uuid4()
    target = SimpleNamespace(weights={instrument_id: Decimal("0.25")})
    session = _session(target=target)
    strategy_run = _strategy_run()

    result = await run_shadow_target_cycle(
        session=session,
        strategy_run=strategy_run,
        feature_data={instrument_id: {"momentum": 1.0}},
        market_prices={instrument_id: Decimal("100")},
        engine_name="equity",
    )

    session.signal_ctrl.generate.assert_awaited_once_with(
        feature_data={instrument_id: {"momentum": 1.0}},
        strategy_run=strategy_run,
        as_of=_AS_OF,
    )
    session.portfolio_ctrl.build.assert_awaited_once()
    session.order_planner.plan.assert_called_once()
    session.approve_ctrl.approve.assert_awaited_once_with(
        ["intent"], session.account_broker.sync_account.return_value
    )
    assert result.signals == ["signal"]
    assert result.target is target
    assert result.approved == ["approved"]
    assert result.rejected == ["rejected"]
    assert result.submitted_ids == []
    assert result.fills == []
    session.performance_repo.save_nav_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_shadow_target_cycle_skips_planning_when_no_target() -> None:
    session = _session(target=None)

    result = await run_shadow_target_cycle(
        session=session,
        strategy_run=_strategy_run(),
        feature_data={},
        market_prices=None,
        engine_name="equity",
    )

    session.order_planner.plan.assert_not_called()
    session.approve_ctrl.approve.assert_not_called()
    assert result.target is None
    assert result.approved == []
    assert result.rejected == []


@pytest.mark.asyncio
async def test_shadow_nav_snapshot_is_best_effort() -> None:
    session = SimpleNamespace(
        performance_repo=SimpleNamespace(
            save_nav_snapshot=AsyncMock(side_effect=RuntimeError("offline"))
        )
    )

    await save_shadow_nav_snapshot(
        session=session,
        strategy_run=_strategy_run(),
        account=_account(),
    )

    session.performance_repo.save_nav_snapshot.assert_awaited_once()
