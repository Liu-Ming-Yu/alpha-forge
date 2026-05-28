"""Unit tests for shadow boosting-cycle orchestration."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.shadow.boosting_cycle import run_shadow_boosting_cycle

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_shadow_boosting_cycle_noops_without_scorer() -> None:
    await run_shadow_boosting_cycle(
        scorer=None,
        strategy_run=object(),
        feature_data={uuid.uuid4(): {"momentum": 1.0}},
        primary_scores=[object()],
        as_of=_AS_OF,
        engine_name="equity",
    )


@pytest.mark.asyncio
async def test_shadow_boosting_cycle_scores_features(tmp_path) -> None:
    scorer = AsyncMock()
    scorer.score_cycle = AsyncMock(return_value=tmp_path / "boosting.jsonl")
    strategy_run = object()
    instrument_id = uuid.uuid4()

    await run_shadow_boosting_cycle(
        scorer=scorer,
        strategy_run=strategy_run,
        feature_data={instrument_id: {"momentum": 1.0}},
        primary_scores=["score"],
        as_of=_AS_OF,
        engine_name="equity",
    )

    scorer.score_cycle.assert_awaited_once_with(
        feature_data={instrument_id: {"momentum": 1.0}},
        primary_scores=["score"],
        strategy_run=strategy_run,
        as_of=_AS_OF,
    )


@pytest.mark.asyncio
async def test_shadow_boosting_cycle_swallows_scorer_failure() -> None:
    scorer = AsyncMock()
    scorer.score_cycle = AsyncMock(side_effect=RuntimeError("offline"))

    await run_shadow_boosting_cycle(
        scorer=scorer,
        strategy_run=object(),
        feature_data={},
        primary_scores=[],
        as_of=_AS_OF,
        engine_name="equity",
    )

    scorer.score_cycle.assert_awaited_once()
