"""Unit tests for shadow text-cycle orchestration."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.shadow.text_cycle import run_shadow_text_cycle

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


def _event(artifact_uri: str) -> SimpleNamespace:
    return SimpleNamespace(event_id=uuid.uuid4(), artifact_uri=artifact_uri)


def _strategy_run() -> SimpleNamespace:
    return SimpleNamespace(strategy_name="equity")


@pytest.mark.asyncio
async def test_shadow_text_cycle_noops_without_scorer() -> None:
    session = SimpleNamespace(text_event_store=AsyncMock())

    await run_shadow_text_cycle(
        scorer=None,
        session=session,
        strategy_run=_strategy_run(),
        as_of=_AS_OF,
    )

    session.text_event_store.get_events.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_text_cycle_reads_artifacts_and_scores_events(tmp_path) -> None:
    artifact = tmp_path / "text.txt"
    artifact.write_text("guidance raised", encoding="utf-8")
    event = _event(str(artifact))
    scorer = AsyncMock()
    scorer.score_cycle = AsyncMock()
    session = SimpleNamespace(
        text_event_store=AsyncMock(get_events=AsyncMock(return_value=[event])),
        performance_repo=None,
    )
    market_prices = {uuid.uuid4(): Decimal("100")}

    await run_shadow_text_cycle(
        scorer=scorer,
        session=session,
        strategy_run=_strategy_run(),
        as_of=_AS_OF,
        market_prices=market_prices,
    )

    scorer.score_cycle.assert_awaited_once()
    kwargs = scorer.score_cycle.await_args.kwargs
    assert kwargs["events"] == [event]
    assert kwargs["text_contents"] == {event.event_id: "guidance raised"}
    assert kwargs["market_prices"] == market_prices


@pytest.mark.asyncio
async def test_shadow_text_cycle_skips_when_ic_gate_is_warmed_up_and_failing(tmp_path) -> None:
    event = _event(str(tmp_path / "missing.txt"))
    scorer = AsyncMock()
    scorer.score_cycle = AsyncMock()
    session = SimpleNamespace(
        text_event_store=AsyncMock(get_events=AsyncMock(return_value=[event])),
        performance_repo=AsyncMock(
            status=AsyncMock(
                return_value=SimpleNamespace(
                    observations=25,
                    min_observations=20,
                    passed=False,
                    rolling_ic=-0.02,
                    negative_streak=4,
                )
            )
        ),
    )

    await run_shadow_text_cycle(
        scorer=scorer,
        session=session,
        strategy_run=_strategy_run(),
        as_of=_AS_OF,
    )

    session.performance_repo.status.assert_awaited_once()
    session.text_event_store.get_events.assert_not_called()
    scorer.score_cycle.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_text_cycle_continues_when_ic_gate_check_fails(tmp_path) -> None:
    artifact = tmp_path / "text.txt"
    artifact.write_text("margin expanded", encoding="utf-8")
    event = _event(str(artifact))
    scorer = AsyncMock()
    scorer.score_cycle = AsyncMock()
    session = SimpleNamespace(
        text_event_store=AsyncMock(get_events=AsyncMock(return_value=[event])),
        performance_repo=AsyncMock(status=AsyncMock(side_effect=RuntimeError("offline"))),
    )

    await run_shadow_text_cycle(
        scorer=scorer,
        session=session,
        strategy_run=_strategy_run(),
        as_of=_AS_OF,
    )

    scorer.score_cycle.assert_awaited_once()
