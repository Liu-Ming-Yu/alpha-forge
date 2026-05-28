"""Intraday candidate screening orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.campaigns.screening.common import (
    build_screen_candidate_row,
    candidate_screen_threshold_payload,
    ranked_candidate_diagnostics,
    sample_build_summary,
)
from quant_platform.services.research_service.intraday.candidates.features import (
    IntradayCandidateFeatureSpec,
    attach_intraday_candidate_features,
    intraday_source_summary,
)
from quant_platform.services.research_service.intraday.candidates.screening.candidates import (
    INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
)
from quant_platform.services.research_service.intraday.candidates.screening.types import (
    INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION,
    IntradayCandidateScreenThresholds,
    IntradayCandidateSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def build_intraday_candidate_screen(
    *,
    samples: Sequence[SupervisedAlphaSample],
    intraday_bars: Sequence[MarketBar],
    sample_build: Mapping[str, object],
    intraday_feature_set_version: str = INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION,
    candidate_family: str = "intraday-microstructure-v1",
    thresholds: IntradayCandidateScreenThresholds | None = None,
    seed: int = 17,
    permutation_count: int = 200,
    candidate_set: str = "seed",
    candidates: Sequence[IntradayCandidateSpec] = INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES,
) -> dict[str, object]:
    """Screen 1-minute microstructure candidates before registering feature metadata."""
    active_thresholds = thresholds or IntradayCandidateScreenThresholds()
    source_summary = intraday_source_summary(intraday_bars=intraday_bars, samples=samples)
    build_summary = sample_build_summary(sample_build, sample_count=len(samples))
    global_blockers = [
        *list(cast("Sequence[object]", source_summary["blockers"])),
        *list(cast("Sequence[object]", build_summary["blockers"])),
    ]
    screened_samples = attach_intraday_candidate_features(
        samples=samples,
        intraday_bars=intraday_bars,
        candidates=cast("Sequence[IntradayCandidateFeatureSpec]", candidates),
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
            "intraday candidate screen admitted "
            f"{len(passing)} features, required {active_thresholds.min_passing_candidates}"
        )
    return {
        "passed": not blockers,
        "reason": "intraday candidate screen passed"
        if not blockers
        else "intraday candidate screen blocked prospective feature family",
        "blockers": blockers,
        "diagnostic_only": True,
        "promotion_artifacts_written": False,
        "candidate_family": candidate_family,
        "candidate_set": candidate_set,
        "intraday_feature_set_version": intraday_feature_set_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "permutation_seed": seed,
        "permutation_count": permutation_count,
        "thresholds": candidate_screen_threshold_payload(active_thresholds),
        "intraday_source_validation": source_summary,
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
