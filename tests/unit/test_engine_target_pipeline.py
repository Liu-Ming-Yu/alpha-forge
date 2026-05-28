"""Unit tests for the shared engine signal-to-target pipeline."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.proposals.target_pipeline import build_engine_target

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_build_engine_target_syncs_account_scores_regime_and_target() -> None:
    account = object()
    regime = SimpleNamespace(regime_label=SimpleNamespace(value="risk_on"))
    target = object()
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())
    feature_data = {uuid.uuid4(): {"momentum": 1.0}}
    session = SimpleNamespace(
        account_broker=SimpleNamespace(sync_account=AsyncMock(return_value=account)),
        signal_ctrl=SimpleNamespace(generate=AsyncMock(return_value=("signal",))),
        regime_detector=SimpleNamespace(detect=AsyncMock(return_value=regime)),
        portfolio_ctrl=SimpleNamespace(build=AsyncMock(return_value=target)),
        risk_limits=object(),
    )

    result = await build_engine_target(
        session=session,
        strategy_run=strategy_run,
        feature_data=feature_data,
        as_of=_AS_OF,
    )

    session.account_broker.sync_account.assert_awaited_once()
    session.signal_ctrl.generate.assert_awaited_once_with(
        feature_data=feature_data,
        strategy_run=strategy_run,
        as_of=_AS_OF,
    )
    session.regime_detector.detect.assert_awaited_once_with(_AS_OF)
    session.portfolio_ctrl.build.assert_awaited_once_with(
        signals=("signal",),
        regime=regime,
        account=account,
        limits=session.risk_limits,
    )
    assert result.account is account
    assert result.signals == ["signal"]
    assert result.regime is regime
    assert result.target is target
