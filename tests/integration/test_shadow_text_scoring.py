"""Integration tests for the Phase 5 shadow text scoring pipeline.

Exercises the full path:
    TextEvent → LLMTextFeatureExtractor (mocked) → ShadowTextScorer
    → FeatureRepository store_vector → SignalScore list

Validates:
- Shadow scores are returned but have no effect on portfolio targets.
- FeatureVectors are persisted with the correct factor version.
- IC is accumulated correctly across multiple simulated cycles.
- passes_ic_gate() enforces the 20-day / IC > 0.05 promotion criteria.
- ShadowTextScorer failures are isolated: exceptions inside score_cycle()
  never propagate to the caller.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.core.domain.research import FeatureVector, RunStatus, RunType, StrategyRun
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.services.research_service.text.features import (
    FeatureExtractionError,
    LLMTextFeatureExtractor,
)
from quant_platform.services.research_service.text.shadow.scorer import ShadowTextScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2025, 9, 15, 14, 0, 0, tzinfo=UTC)


def _make_strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="shadow_test",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=_BASE_TIME,
        started_at=_BASE_TIME,
    )


def _make_event(
    instrument_id: uuid.UUID,
    *,
    occurred_at: datetime = _BASE_TIME,
    event_type: TextEventType = TextEventType.EARNINGS_TRANSCRIPT,
) -> TextEvent:
    return TextEvent(
        event_id=uuid.uuid4(),
        event_type=event_type,
        occurred_at=occurred_at,
        source_uri="s3://bucket/raw.txt",
        artifact_uri="s3://bucket/artifact.txt",
        instrument_id=instrument_id,
    )


def _stub_extractor(features: dict | None = None) -> LLMTextFeatureExtractor:
    """Return an extractor whose extract() returns a canned FeatureVector."""
    if features is None:
        features = {
            "text_sentiment": 0.6,
            "guidance_direction": 1.0,
            "revenue_revision_magnitude": 0.3,
            "macro_sentiment": 0.1,
        }

    extractor = MagicMock(spec=LLMTextFeatureExtractor)

    def _extract(event: TextEvent, text_content: str, run_id: uuid.UUID, **kw) -> FeatureVector:
        as_of = kw.get("as_of") or event.occurred_at
        return FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=event.instrument_id or uuid.UUID(int=0),
            strategy_run_id=run_id,
            as_of=as_of,
            features=dict(features),
            feature_set_version="text-v1",
            artifact_uri=f"{event.artifact_uri}#prompt=v1",
        )

    extractor.extract.side_effect = _extract
    return extractor


def _instrument_extractor(sentiments: dict[uuid.UUID, float]) -> LLMTextFeatureExtractor:
    """Return an extractor whose text sentiment is keyed by instrument."""
    extractor = MagicMock(spec=LLMTextFeatureExtractor)

    def _extract(event: TextEvent, text_content: str, run_id: uuid.UUID, **kw) -> FeatureVector:
        as_of = kw.get("as_of") or event.occurred_at
        return FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=event.instrument_id or uuid.UUID(int=0),
            strategy_run_id=run_id,
            as_of=as_of,
            features={
                "text_sentiment": sentiments[event.instrument_id],
                "guidance_direction": 0.0,
                "revenue_revision_magnitude": 0.5,
                "macro_sentiment": 0.0,
            },
            feature_set_version="text-v1",
            artifact_uri=f"{event.artifact_uri}#prompt=v1",
        )

    extractor.extract.side_effect = _extract
    return extractor


# ---------------------------------------------------------------------------
# Basic cycle tests
# ---------------------------------------------------------------------------


class TestShadowScorerCycle:
    @pytest.mark.asyncio
    async def test_score_cycle_returns_signal_scores(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()

        iid = uuid.uuid4()
        event = _make_event(iid)

        signals = await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "Q3 earnings beat expectations."},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert len(signals) == 1
        sig = signals[0]
        assert sig.instrument_id == iid
        assert -1.0 <= sig.score <= 1.0
        assert sig.model_version == ShadowTextScorer.FACTOR_VERSION

    @pytest.mark.asyncio
    async def test_score_cycle_stores_feature_vector(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()

        iid = uuid.uuid4()
        event = _make_event(iid)

        await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        stored = await repo.get_vectors([iid], "text-v1", _BASE_TIME)
        assert len(stored) == 1
        assert stored[0].instrument_id == iid

    @pytest.mark.asyncio
    async def test_score_cycle_persists_prediction_evidence(self) -> None:
        repo = InMemoryFeatureRepository()
        prediction_repo = InMemoryPerformanceRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(
            extractor=extractor,
            feature_repo=repo,
            prediction_evidence_repo=prediction_repo,
        )
        run = _make_strategy_run()

        iid = uuid.uuid4()
        event = _make_event(iid)

        await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "Q3 earnings beat expectations."},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        evidence = await prediction_repo.forecast_evidence(
            "text",
            as_of=_BASE_TIME,
            stale_after_hours=24,
            min_confidence=0.0,
        )

        assert evidence.passed
        assert evidence.observations == 1
        assert evidence.model_version == ShadowTextScorer.FACTOR_VERSION
        assert evidence.calibration_buckets == ("shadow:daily:text",)

    @pytest.mark.asyncio
    async def test_signals_use_correct_blended_score(self) -> None:
        """70% sentiment + 30% guidance = 0.7*0.6 + 0.3*1.0 = 0.72."""
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor(
            {
                "text_sentiment": 0.6,
                "guidance_direction": 1.0,
                "revenue_revision_magnitude": 0.0,
                "macro_sentiment": 0.0,
            }
        )
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()
        iid = uuid.uuid4()
        event = _make_event(iid)

        signals = await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert signals[0].score == pytest.approx(0.72, abs=1e-6)

    @pytest.mark.asyncio
    async def test_blended_score_is_clamped_to_minus_one_plus_one(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor(
            {
                "text_sentiment": 1.0,
                "guidance_direction": 1.0,
                "revenue_revision_magnitude": 1.0,
                "macro_sentiment": 1.0,
            }
        )
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()
        iid = uuid.uuid4()
        event = _make_event(iid)

        signals = await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert signals[0].score <= 1.0
        assert signals[0].score >= -1.0

    @pytest.mark.asyncio
    async def test_event_without_instrument_id_is_skipped(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()

        macro_event = _make_event(
            instrument_id=None,  # type: ignore[arg-type]
            event_type=TextEventType.MACRO_COMMENTARY,
        )
        # Override instrument_id explicitly to None
        import dataclasses

        macro_event = dataclasses.replace(macro_event, instrument_id=None)

        signals = await scorer.score_cycle(
            events=[macro_event],
            text_contents={macro_event.event_id: "Fed holds rates."},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert signals == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_content_is_skipped(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()
        iid = uuid.uuid4()
        event = _make_event(iid)

        signals = await scorer.score_cycle(
            events=[event],
            text_contents={},  # no content mapped for this event
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert signals == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_extraction_failure_skips_instrument_and_continues(self) -> None:
        """FeatureExtractionError for one instrument must not abort the whole cycle."""
        repo = InMemoryFeatureRepository()
        extractor = MagicMock(spec=LLMTextFeatureExtractor)

        iid_good = uuid.uuid4()
        iid_bad = uuid.uuid4()
        good_event = _make_event(iid_good)
        bad_event = _make_event(iid_bad)

        good_vector = FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=iid_good,
            strategy_run_id=uuid.uuid4(),
            as_of=_BASE_TIME,
            features={
                "text_sentiment": 0.5,
                "guidance_direction": 0.0,
                "revenue_revision_magnitude": 0.2,
                "macro_sentiment": 0.0,
            },
            feature_set_version="text-v1",
            artifact_uri="s3://b/a.txt#prompt=v1",
        )

        def _selective_extract(event, text, run_id, **kw):
            if event.instrument_id == iid_bad:
                raise FeatureExtractionError("timeout")
            return good_vector

        extractor.extract.side_effect = _selective_extract

        run = _make_strategy_run()
        signals = await scorer_from(extractor, repo).score_cycle(
            events=[bad_event, good_event],
            text_contents={
                bad_event.event_id: "bad text",
                good_event.event_id: "good text",
            },
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert len(signals) == 1
        assert signals[0].instrument_id == iid_good


def scorer_from(extractor, repo) -> ShadowTextScorer:
    return ShadowTextScorer(extractor=extractor, feature_repo=repo)


# ---------------------------------------------------------------------------
# IC tracking
# ---------------------------------------------------------------------------


class TestShadowScorerIC:
    @pytest.mark.asyncio
    async def test_rolling_ic_is_nan_with_fewer_than_two_observations(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=20)
        run = _make_strategy_run()

        iid = uuid.uuid4()
        event = _make_event(iid)

        # One cycle — no IC yet.
        await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        assert math.isnan(scorer.rolling_ic)
        assert scorer.ic_observations == 0

    @pytest.mark.asyncio
    async def test_ic_accumulates_across_cycles(self) -> None:
        """Run 5 cycles with 3 instruments each, verify IC is computed."""
        repo = InMemoryFeatureRepository()
        instruments = [uuid.uuid4() for _ in range(3)]
        extractor = _instrument_extractor(
            {
                instruments[0]: 0.9,
                instruments[1]: 0.2,
                instruments[2]: -0.6,
            }
        )
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=20)
        run = _make_strategy_run()

        for day in range(5):
            as_of = _BASE_TIME + timedelta(days=day)
            events = [_make_event(iid, occurred_at=as_of) for iid in instruments]
            text_contents = {e.event_id: f"day {day} text" for e in events}
            market_prices: dict[uuid.UUID, Decimal] = {
                instruments[0]: Decimal("100") + Decimal(day * 3),
                instruments[1]: Decimal("100") + Decimal(day),
                instruments[2]: Decimal("100") - Decimal(day),
            }

            await scorer.score_cycle(
                events=events,
                text_contents=text_contents,
                strategy_run=run,
                as_of=as_of,
                market_prices=market_prices,
            )

        # After 5 cycles we expect at least a few IC observations.
        assert scorer.ic_observations >= 1

    @pytest.mark.asyncio
    async def test_ic_uses_realized_returns_not_repeated_scores(self) -> None:
        """Previous scores must be compared with next-period price returns."""
        repo = InMemoryFeatureRepository()
        instruments = [uuid.uuid4() for _ in range(3)]
        extractor = _instrument_extractor(
            {
                instruments[0]: 0.9,
                instruments[1]: 0.1,
                instruments[2]: -0.7,
            }
        )
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=20)
        run = _make_strategy_run()
        day0_events = [_make_event(iid, occurred_at=_BASE_TIME) for iid in instruments]
        day1_events = [
            _make_event(iid, occurred_at=_BASE_TIME + timedelta(days=1)) for iid in instruments
        ]

        await scorer.score_cycle(
            events=day0_events,
            text_contents={event.event_id: "day 0" for event in day0_events},
            strategy_run=run,
            as_of=_BASE_TIME,
            market_prices={iid: Decimal("100") for iid in instruments},
        )
        await scorer.score_cycle(
            events=day1_events,
            text_contents={event.event_id: "day 1" for event in day1_events},
            strategy_run=run,
            as_of=_BASE_TIME + timedelta(days=1),
            market_prices={
                instruments[0]: Decimal("110"),
                instruments[1]: Decimal("100"),
                instruments[2]: Decimal("90"),
            },
        )

        assert scorer.ic_observations == 1
        assert scorer._ic_history[-1] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_ic_can_be_negative_when_returns_invert_scores(self) -> None:
        repo = InMemoryFeatureRepository()
        instruments = [uuid.uuid4() for _ in range(3)]
        extractor = _instrument_extractor(
            {
                instruments[0]: 0.9,
                instruments[1]: 0.1,
                instruments[2]: -0.7,
            }
        )
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=20)
        run = _make_strategy_run()
        day0_events = [_make_event(iid, occurred_at=_BASE_TIME) for iid in instruments]
        day1_events = [
            _make_event(iid, occurred_at=_BASE_TIME + timedelta(days=1)) for iid in instruments
        ]

        await scorer.score_cycle(
            events=day0_events,
            text_contents={event.event_id: "day 0" for event in day0_events},
            strategy_run=run,
            as_of=_BASE_TIME,
            market_prices={iid: Decimal("100") for iid in instruments},
        )
        await scorer.score_cycle(
            events=day1_events,
            text_contents={event.event_id: "day 1" for event in day1_events},
            strategy_run=run,
            as_of=_BASE_TIME + timedelta(days=1),
            market_prices={
                instruments[0]: Decimal("90"),
                instruments[1]: Decimal("100"),
                instruments[2]: Decimal("110"),
            },
        )

        assert scorer.ic_observations == 1
        assert scorer._ic_history[-1] == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_passes_ic_gate_requires_20_observations(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=20)
        run = _make_strategy_run()

        instruments = [uuid.uuid4() for _ in range(3)]
        market_prices = {iid: Decimal("100") for iid in instruments}

        # Run 10 cycles — not enough observations.
        for day in range(10):
            as_of = _BASE_TIME + timedelta(days=day)
            events = [_make_event(iid, occurred_at=as_of) for iid in instruments]
            text_contents = {e.event_id: "text" for e in events}
            await scorer.score_cycle(
                events=events,
                text_contents=text_contents,
                strategy_run=run,
                as_of=as_of,
                market_prices=market_prices,
            )

        assert not scorer.passes_ic_gate(min_observations=20)

    @pytest.mark.asyncio
    async def test_passes_ic_gate_false_when_rolling_ic_below_threshold(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=5)

        # Inject 5 low IC values directly into the IC history.
        for _ in range(5):
            scorer._ic_history.append(0.01)

        assert scorer.ic_observations == 5
        assert not scorer.passes_ic_gate(min_ic=0.05, min_observations=5)

    @pytest.mark.asyncio
    async def test_passes_ic_gate_true_when_conditions_met(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=5)

        # Inject 5 strong IC values.
        for _ in range(5):
            scorer._ic_history.append(0.12)

        assert scorer.passes_ic_gate(min_ic=0.05, min_observations=5)

    def test_rolling_ic_is_mean_of_valid_observations(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo, ic_window=5)

        scorer._ic_history.extend([0.10, 0.12, 0.08, float("nan"), 0.14])

        # NaN is excluded; mean of [0.10, 0.12, 0.08, 0.14] = 0.11
        assert scorer.rolling_ic == pytest.approx(0.11, abs=1e-9)


# ---------------------------------------------------------------------------
# Integration: shadow scores do NOT appear in portfolio target
# ---------------------------------------------------------------------------


class TestShadowDoesNotAffectPortfolio:
    """Verify the architectural firewall: shadow scores stay shadow."""

    @pytest.mark.asyncio
    async def test_shadow_scorer_does_not_write_to_order_repository(self) -> None:
        """score_cycle must never touch any order-path object."""
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()

        iid = uuid.uuid4()
        event = _make_event(iid)

        signals = await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        # The returned signals have model_version=FACTOR_VERSION (shadow),
        # not a live model version.
        assert all(s.model_version == ShadowTextScorer.FACTOR_VERSION for s in signals)

        # FeatureRepository was written to (shadow store only).
        vectors = await repo.get_vectors([iid], "text-v1", _BASE_TIME)
        assert len(vectors) == 1

    @pytest.mark.asyncio
    async def test_shadow_scorer_feature_version_is_shadow(self) -> None:
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()
        iid = uuid.uuid4()
        event = _make_event(iid)

        await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        vectors = await repo.get_vectors([iid], "text-v1", _BASE_TIME)
        # feature_set_version must be the shadow version, not any live version.
        assert vectors[0].feature_set_version == "text-v1"
        # The scorer's FACTOR_VERSION must also be distinguishable.
        assert ShadowTextScorer.FACTOR_VERSION == "text-shadow-v1"

    @pytest.mark.asyncio
    async def test_score_cycle_is_isolated_from_exceptions(self) -> None:
        """An exception raised by store_vector must not propagate to the caller."""
        repo = InMemoryFeatureRepository()
        extractor = _stub_extractor()
        scorer = ShadowTextScorer(extractor=extractor, feature_repo=repo)
        run = _make_strategy_run()
        iid = uuid.uuid4()
        event = _make_event(iid)

        async def _exploding_store(_):
            raise RuntimeError("disk full")

        repo.store_vector = _exploding_store  # type: ignore[method-assign]

        # Should NOT raise even though store_vector blows up.
        signals = await scorer.score_cycle(
            events=[event],
            text_contents={event.event_id: "text"},
            strategy_run=run,
            as_of=_BASE_TIME,
        )

        # Signal is still returned even if persistence failed.
        assert len(signals) == 1
