"""Event candidate screening orchestration."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.campaigns.screening.common import (
    build_screen_candidate_row,
    candidate_screen_threshold_payload,
    ranked_candidate_diagnostics,
    sample_build_summary,
    sec_source_manifest_summary,
)
from quant_platform.services.research_service.events.candidates.screening.candidates import (
    EVENT_REACTION_SEED_CANDIDATES,
)
from quant_platform.services.research_service.events.candidates.screening.context import (
    _event_context_features,
    _events_by_instrument,
)
from quant_platform.services.research_service.events.candidates.screening.types import (
    EVENT_REACTION_FEATURE_SET_VERSION,
    EventCandidateScreenThresholds,
    EventCandidateSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def build_event_candidate_screen(
    *,
    samples: Sequence[SupervisedAlphaSample],
    source_manifest: Mapping[str, object],
    sample_build: Mapping[str, object],
    event_feature_set_version: str = EVENT_REACTION_FEATURE_SET_VERSION,
    candidate_family: str = "event-reaction-v1",
    thresholds: EventCandidateScreenThresholds | None = None,
    seed: int = 17,
    permutation_count: int = 200,
    candidate_set: str = "seed",
    candidates: Sequence[EventCandidateSpec] = EVENT_REACTION_SEED_CANDIDATES,
) -> dict[str, object]:
    """Screen event/price-reaction candidates before registering a feature set."""
    active_thresholds = thresholds or EventCandidateScreenThresholds()
    source_summary = sec_source_manifest_summary(source_manifest)
    build_summary = sample_build_summary(sample_build, sample_count=len(samples))
    global_blockers = [
        *list(cast("Sequence[object]", source_summary["blockers"])),
        *list(cast("Sequence[object]", build_summary["blockers"])),
    ]
    screened_samples = _with_event_features(
        samples=samples,
        candidates=candidates,
        source_manifest=source_manifest,
    )
    rows = [
        build_screen_candidate_row(
            samples=screened_samples,
            feature_name=candidate.name,
            expression=candidate.expression,
            thresholds=active_thresholds,
            seed=seed,
            permutation_count=permutation_count,
        )
        for candidate in candidates
    ]
    passing = tuple(str(row["feature_name"]) for row in rows if bool(row["passed"]))
    ranked = ranked_candidate_diagnostics(rows)
    blockers = [str(blocker) for blocker in global_blockers]
    if len(passing) < active_thresholds.min_passing_candidates:
        blockers.append(
            "event candidate screen admitted "
            f"{len(passing)} features, required {active_thresholds.min_passing_candidates}"
        )
    return {
        "passed": not blockers,
        "reason": "event candidate screen passed"
        if not blockers
        else "event candidate screen blocked prospective feature family",
        "blockers": blockers,
        "diagnostic_only": True,
        "promotion_artifacts_written": False,
        "candidate_family": candidate_family,
        "candidate_set": candidate_set,
        "event_feature_set_version": event_feature_set_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "permutation_seed": seed,
        "permutation_count": permutation_count,
        "thresholds": candidate_screen_threshold_payload(active_thresholds),
        "source_manifest_validation": source_summary,
        "sample_build_validation": build_summary,
        "sample_count": len(samples),
        "screened_candidate_count": len(rows),
        "passing_candidates": list(passing),
        "quarantined_candidates": [
            str(row["feature_name"]) for row in rows if not bool(row["passed"])
        ],
        "top_by_stability": ranked["top_by_stability"],
        "top_by_null_margin": ranked["top_by_null_margin"],
        "near_misses": ranked["near_misses"],
        "candidates": rows,
    }


def _with_event_features(
    *,
    samples: Sequence[SupervisedAlphaSample],
    candidates: Sequence[EventCandidateSpec],
    source_manifest: Mapping[str, object] | None = None,
) -> tuple[SupervisedAlphaSample, ...]:
    events_by_instrument = _events_by_instrument(source_manifest or {})
    screened: list[SupervisedAlphaSample] = []
    for sample in samples:
        features = dict(sample.features)
        formula_row = {
            **sample.features,
            **_event_context_features(sample, events_by_instrument),
        }
        for candidate in candidates:
            features[candidate.name] = candidate.formula(formula_row)
        screened.append(dataclasses.replace(sample, features=features))
    return tuple(screened)
