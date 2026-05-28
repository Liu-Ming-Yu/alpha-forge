"""Unit tests for engine proposal generation helper."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.framework.types import RunMode
from quant_platform.engines.proposals.generation import generate_engine_target_proposal

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


def _session(*, target: object | None) -> SimpleNamespace:
    regime = SimpleNamespace(regime_label=SimpleNamespace(value="risk_on"))
    return SimpleNamespace(
        account_broker=SimpleNamespace(sync_account=AsyncMock(return_value=object())),
        signal_ctrl=SimpleNamespace(generate=AsyncMock(return_value=[])),
        regime_detector=SimpleNamespace(detect=AsyncMock(return_value=regime)),
        portfolio_ctrl=SimpleNamespace(build=AsyncMock(return_value=target)),
        risk_limits=object(),
    )


@pytest.mark.asyncio
async def test_generate_engine_target_proposal_returns_target_payload() -> None:
    instrument_id = uuid.uuid4()
    feature_dataset_id = uuid.uuid4()
    target = SimpleNamespace(
        as_of=_AS_OF,
        weights={instrument_id: Decimal("0.20")},
        cash_target_weight=Decimal("0.80"),
        construction_notes=["ok"],
    )
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())

    proposal = await generate_engine_target_proposal(
        session=_session(target=target),
        strategy_run=strategy_run,
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.PAPER,
        feature_data={instrument_id: {"momentum": 1.0}},
        as_of=_AS_OF,
        feature_dataset_id=feature_dataset_id,
    )

    assert proposal.engine_name == "equity"
    assert proposal.weights == {instrument_id: Decimal("0.20")}
    assert proposal.cash_target_weight == Decimal("0.80")
    assert proposal.feature_dataset_id == feature_dataset_id
    assert proposal.notes == ("ok",)


@pytest.mark.asyncio
async def test_generate_engine_target_proposal_returns_all_cash_rejection() -> None:
    strategy_run = SimpleNamespace(run_id=uuid.uuid4())

    proposal = await generate_engine_target_proposal(
        session=_session(target=None),
        strategy_run=strategy_run,
        engine_name="equity",
        engine_version="1.2.3",
        run_mode=RunMode.LIVE,
        feature_data={},
        as_of=_AS_OF,
    )

    assert proposal.weights == {}
    assert proposal.cash_target_weight == Decimal("1")
    assert proposal.promotion_state == "live"
    assert proposal.notes == ("risk_policy_rejected",)
