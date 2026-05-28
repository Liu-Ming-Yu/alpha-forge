from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.services.research_service.events.candidates.screening import (
    EVENT_REACTION_FEATURE_SET_VERSION,
    EVENT_REACTION_V2_FEATURE_SET_VERSION,
    EventCandidateScreenThresholds,
    build_event_candidate_screen,
    event_candidates_for_set,
    write_event_candidate_family_artifacts,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path


def test_event_candidate_screen_writes_family_only_after_pass(tmp_path: Path) -> None:
    samples = _event_samples()
    screen = build_event_candidate_screen(
        samples=samples,
        source_manifest=_manifest(),
        sample_build=_sample_build(len(samples)),
        thresholds=EventCandidateScreenThresholds(min_passing_candidates=3),
        permutation_count=30,
    )

    assert screen["passed"] is True
    assert len(screen["passing_candidates"]) >= 3
    assert screen["diagnostic_only"] is True
    assert screen["promotion_artifacts_written"] is False

    written = write_event_candidate_family_artifacts(
        screen=screen,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path / "feature_families" / "paper-alpha-event-reaction-v1.json",
    )

    assert written["written"] is True
    assert (tmp_path / "feature_families" / "paper-alpha-event-reaction-v1.json").exists()
    assert len(list((tmp_path / "feature_cards").glob("*.json"))) >= 3


def test_event_candidate_screen_blocks_sparse_or_unstable_candidates(tmp_path: Path) -> None:
    samples = _event_samples(zero_features=True)
    screen = build_event_candidate_screen(
        samples=samples,
        source_manifest=_manifest(),
        sample_build=_sample_build(len(samples)),
        permutation_count=20,
    )

    assert screen["passed"] is False
    assert "event candidate screen admitted 0 features" in " ".join(screen["blockers"])

    written = write_event_candidate_family_artifacts(
        screen=screen,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path / "feature_families" / "paper-alpha-event-reaction-v1.json",
    )

    assert written == {"written": False, "reason": "screen did not pass"}
    assert not (tmp_path / "feature_cards").exists()


def test_event_candidate_sets_are_deterministic() -> None:
    candidates = event_candidates_for_set("seed")

    assert len(candidates) == 5
    assert [candidate.name for candidate in candidates] == [
        "abnormal_volume_3d_decay",
        "event_gap_reversal_1d_decay",
        "post_event_drift_confirmation_3d_decay",
        "filing_cadence_surprise_decay",
        "event_attention_shock_30d_decay",
    ]
    assert EVENT_REACTION_FEATURE_SET_VERSION == "paper-alpha-event-reaction-v1"

    v2 = event_candidates_for_set("event-reaction-v2")
    assert [candidate.name for candidate in v2] == [
        "event_reaction_v2_sec_density_price_reversal_21d",
        "event_reaction_v2_attention_gap_reversal_21d",
        "event_reaction_v2_post_event_drift_quality_21d",
        "event_reaction_v2_extreme_attention_reversal_21d",
        "event_reaction_v2_crowded_medium_momentum_decay_21d",
        "event_reaction_v2_sec_count_1_4_momo1_extreme_21d",
        "event_reaction_v2_sec_count_2_5_momo1_extreme_21d",
        "event_reaction_v2_sec_count_3_5_momo1_medium_momo_21d",
        "event_reaction_v2_sec_count_3_6_momo1_reversal_21d",
        "event_reaction_v2_sec_count_7_9_momo1_momo3_21d",
        "event_reaction_v2_sec_count_7_9_momo1_trend_21d",
        "event_reaction_v2_sec_count_7_9_momo3_trend_21d",
    ]
    assert EVENT_REACTION_V2_FEATURE_SET_VERSION == "paper-alpha-event-reaction-v2"


def _event_samples(*, zero_features: bool = False) -> tuple[SupervisedAlphaSample, ...]:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    instruments = tuple(uuid.uuid4() for _ in range(6))
    samples: list[SupervisedAlphaSample] = []
    for day in range(8):
        as_of = start + timedelta(days=day)
        for rank, instrument_id in enumerate(instruments, start=1):
            signal = 0.0 if zero_features else float(rank)
            samples.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features={
                        "vol_compression": signal,
                        "short_term_reversal_5d": signal,
                        "momentum_1m": signal,
                        "momentum_3m": signal,
                        "trend_quality_63d": 1.0,
                        "distance_to_52w_high": signal,
                    },
                    forward_return=float(rank),
                )
            )
    return tuple(samples)


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
