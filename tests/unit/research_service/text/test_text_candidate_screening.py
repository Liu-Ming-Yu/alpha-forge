"""Current text candidate catalog and screening tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.core.domain.research import FeatureVector
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample
from quant_platform.services.research_service.text.candidates.catalog import (
    TEXT_CATALYST_V10_PROMOTED_CANDIDATES,
    V10_ALPHA_QUALITY_TEXT_CANDIDATES,
    text_candidate_specs_by_name,
    text_candidates_for_set,
)
from quant_platform.services.research_service.text.candidates.promotion import (
    promote_text_candidate_screens,
)
from quant_platform.services.research_service.text.candidates.screening import (
    TextCandidateScreenThresholds,
    build_text_aggregate_features,
    build_text_candidate_screen,
    write_text_candidate_family_artifacts,
)


def test_v10_candidate_set_is_current_catalog() -> None:
    candidates = text_candidates_for_set("v10-alpha-quality")

    assert candidates == V10_ALPHA_QUALITY_TEXT_CANDIDATES
    assert set(TEXT_CATALYST_V10_PROMOTED_CANDIDATES) <= {
        candidate.name for candidate in candidates
    }


def test_text_candidate_lookup_defaults_to_current_catalog() -> None:
    by_name = text_candidate_specs_by_name()

    assert set(TEXT_CATALYST_V10_PROMOTED_CANDIDATES) <= set(by_name)


def test_unknown_text_candidate_set_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown text candidate set"):
        text_candidates_for_set("retired")


def test_multi_window_text_aggregates_are_point_in_time_safe() -> None:
    instrument_id = uuid.uuid4()
    as_of = datetime(2026, 2, 15, tzinfo=UTC)
    past = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=as_of - timedelta(days=30),
        available_at=as_of - timedelta(days=30),
        feature_set_version="text-v5",
        strategy_run_id=uuid.uuid4(),
        features={"text_sentiment": -2.0, "disclosure_specificity": 1.0},
    )
    future_available = FeatureVector(
        vector_id=uuid.uuid4(),
        instrument_id=instrument_id,
        as_of=as_of - timedelta(days=1),
        available_at=as_of + timedelta(days=1),
        feature_set_version="text-v5",
        strategy_run_id=uuid.uuid4(),
        features={"text_sentiment": 99.0, "disclosure_specificity": 1.0},
    )

    aggregate, decay = build_text_aggregate_features((past, future_available), as_of)

    assert decay is None
    assert aggregate == {}

    aggregate, decay = build_text_aggregate_features(
        (past, future_available),
        as_of,
        lookback_days=42,
    )

    assert decay is not None
    assert aggregate["text_event_count_21d"] == 0.0
    assert aggregate["text_event_count_42d"] == 1.0
    assert aggregate["text_sentiment_decayed_mean_42d"] == pytest.approx(-2.0)


def test_v10_alpha_quality_screen_selects_top_three_and_writes_artifacts(tmp_path) -> None:
    samples, vectors = _screen_fixture()

    payload = build_text_candidate_screen(
        samples=samples,
        text_vectors=vectors,
        source_manifest=_manifest(),
        sample_build=_sample_build(len(samples)),
        text_feature_set_version="text-v5",
        thresholds=TextCandidateScreenThresholds(min_passing_candidates=3),
        permutation_count=30,
        candidate_family="paper-alpha-catalyst-v10",
        candidate_set="v10-alpha-quality",
        candidates=text_candidates_for_set("v10-alpha-quality"),
        promoted_feature_set_version="paper-alpha-catalyst-v10",
    )

    assert payload["passed"] is True
    assert payload["candidate_set"] == "v10-alpha-quality"
    assert len(payload["passing_candidates"]) >= 3
    assert len(payload["selected_candidates"]) == 3
    assert payload["promotion_artifacts_written"] is False

    written = write_text_candidate_family_artifacts(
        screen=payload,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path / "feature_families" / "paper-alpha-catalyst-v10.json",
    )

    assert written["written"] is True
    assert written["selected_candidates"] == payload["selected_candidates"]
    assert (tmp_path / "feature_families" / "paper-alpha-catalyst-v10.json").exists()
    assert len(list((tmp_path / "feature_cards").glob("*.json"))) == 3


def test_v10_alpha_quality_blocked_screen_writes_no_artifacts(tmp_path) -> None:
    samples, vectors = _screen_fixture(zero_text=True)

    payload = build_text_candidate_screen(
        samples=samples,
        text_vectors=vectors,
        source_manifest=_manifest(),
        sample_build=_sample_build(len(samples)),
        text_feature_set_version="text-v5",
        permutation_count=5,
        candidate_set="v10-alpha-quality",
        candidates=text_candidates_for_set("v10-alpha-quality"),
        promoted_feature_set_version="paper-alpha-catalyst-v10",
    )

    assert payload["passed"] is False
    assert payload["selected_candidates"] == []

    written = write_text_candidate_family_artifacts(
        screen=payload,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path / "feature_families" / "paper-alpha-catalyst-v10.json",
    )

    assert written == {"written": False, "reason": "screen did not pass"}
    assert not (tmp_path / "feature_cards").exists()


def test_v10_text_promotion_requires_shared_passing_candidates(tmp_path) -> None:
    screen = {
        "passed": True,
        "candidate_set": "v10-alpha-quality",
        "promoted_feature_set_version": "paper-alpha-catalyst-v10",
        "passing_candidates": list(TEXT_CATALYST_V10_PROMOTED_CANDIDATES),
        "selected_candidates": list(TEXT_CATALYST_V10_PROMOTED_CANDIDATES),
    }

    result = promote_text_candidate_screens(
        main_screen=screen,
        confirmation_screen=screen,
        full_screen=screen,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path / "feature_families" / "paper-alpha-catalyst-v10.json",
    )

    assert result["passed"] is True
    assert result["promotion_artifacts_written"] is True
    assert result["shared_passing_candidates"] == list(TEXT_CATALYST_V10_PROMOTED_CANDIDATES)
    assert (tmp_path / "feature_families" / "paper-alpha-catalyst-v10.json").exists()


def _screen_fixture(
    *,
    reverse_returns: bool = False,
    zero_text: bool = False,
) -> tuple[tuple[SupervisedAlphaSample, ...], tuple[FeatureVector, ...]]:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    instruments = tuple(uuid.uuid4() for _ in range(6))
    samples: list[SupervisedAlphaSample] = []
    vectors: list[FeatureVector] = []
    for rank, instrument_id in enumerate(instruments, start=1):
        score = float(rank)
        text_score = 0.0 if zero_text else score
        vectors.append(
            FeatureVector(
                vector_id=uuid.uuid4(),
                instrument_id=instrument_id,
                as_of=start,
                available_at=start,
                feature_set_version="text-v5",
                strategy_run_id=uuid.uuid4(),
                features={
                    "operating_quality": text_score,
                    "margin_resilience": 1.0 if not zero_text else 0.0,
                    "disclosure_specificity": 1.0 if not zero_text else 0.0,
                    "text_sentiment": -text_score,
                    "catalyst_sentiment": -text_score,
                    "event_surprise": -text_score,
                    "forward_outlook": -text_score,
                    "risk_pressure": 0.0,
                },
            )
        )
    for day in range(8):
        as_of = start + timedelta(days=day)
        for rank, instrument_id in enumerate(instruments, start=1):
            forward_return = float(-rank if reverse_returns else rank)
            samples.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features={"trend_quality_63d": 1.0, "vol_compression": 1.0},
                    forward_return=forward_return,
                )
            )
    return tuple(samples), tuple(vectors)


def _manifest() -> dict[str, object]:
    symbols = [f"S{i:02d}" for i in range(15)]
    return {
        "download": {"requested_symbols": symbols},
        "primary_events_by_symbol": {symbol: 3 for symbol in symbols},
    }


def _sample_build(count: int) -> dict[str, object]:
    return {
        "requested_points": count,
        "samples": count,
        "skipped_missing_features": 0,
        "skipped_stale_features": 0,
        "skipped_missing_bars": 0,
        "skipped_invalid_features": 0,
    }
