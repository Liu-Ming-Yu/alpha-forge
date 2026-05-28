"""Read-only screening for prospective text-derived paper alpha candidates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from quant_platform.services.research_service.campaigns.screening.common import (
    CandidateScreenThresholds,
    build_screen_candidate_row,
    candidate_screen_threshold_payload,
    ranked_candidate_diagnostics,
    sample_build_summary,
    sec_source_manifest_summary,
)
from quant_platform.services.research_service.campaigns.screening.reports import (
    render_candidate_screen_report,
)
from quant_platform.services.research_service.text.candidates.catalog import (
    V10_ALPHA_QUALITY_TEXT_CANDIDATES,
    TextCandidateSpec,
)
from quant_platform.services.research_service.text.candidates.screening.features import (
    _with_screened_features,
)
from quant_platform.services.research_service.text.candidates.screening.selection import (
    _selected_passing_candidates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.core.domain.research import FeatureVector
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

TextCandidateScreenThresholds = CandidateScreenThresholds


def build_text_candidate_screen(
    *,
    samples: Sequence[SupervisedAlphaSample],
    text_vectors: Sequence[FeatureVector],
    source_manifest: Mapping[str, object],
    sample_build: Mapping[str, object],
    text_feature_set_version: str,
    candidate_family: str = "paper-alpha-catalyst-v10",
    lookback_days: int = 21,
    thresholds: TextCandidateScreenThresholds | None = None,
    seed: int = 17,
    permutation_count: int = 200,
    candidate_set: str = "v10-alpha-quality",
    candidates: Sequence[TextCandidateSpec] = V10_ALPHA_QUALITY_TEXT_CANDIDATES,
    promoted_feature_set_version: str = "paper-alpha-catalyst-v10",
) -> dict[str, object]:
    """Screen prospective text candidates before immutable daily backfill."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    active_thresholds = thresholds or TextCandidateScreenThresholds()
    source_summary = sec_source_manifest_summary(source_manifest)
    build_summary = sample_build_summary(sample_build, sample_count=len(samples))
    global_blockers = [
        *list(cast("Sequence[object]", source_summary["blockers"])),
        *list(cast("Sequence[object]", build_summary["blockers"])),
    ]
    screened_samples = _with_screened_features(
        samples=samples,
        text_vectors=text_vectors,
        candidates=candidates,
        lookback_days=lookback_days,
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
    selected = _selected_passing_candidates(
        rows,
        count=active_thresholds.min_passing_candidates,
    )
    ranked = ranked_candidate_diagnostics(rows)
    blockers = list(global_blockers)
    if len(passing) < active_thresholds.min_passing_candidates:
        blockers.append(
            "text candidate screen admitted "
            f"{len(passing)} features, required {active_thresholds.min_passing_candidates}"
        )
    return {
        "passed": not blockers,
        "reason": "text candidate screen passed"
        if not blockers
        else "text candidate screen blocked prospective feature family",
        "blockers": blockers,
        "diagnostic_only": True,
        "promotion_artifacts_written": False,
        "candidate_family": candidate_family,
        "candidate_set": candidate_set,
        "text_feature_set_version": text_feature_set_version,
        "promoted_feature_set_version": promoted_feature_set_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "lookback_days": lookback_days,
        "permutation_seed": seed,
        "permutation_count": permutation_count,
        "thresholds": candidate_screen_threshold_payload(active_thresholds),
        "source_manifest_validation": source_summary,
        "sample_build_validation": build_summary,
        "sample_count": len(samples),
        "text_vector_count": len(text_vectors),
        "screened_candidate_count": len(rows),
        "passing_candidates": list(passing),
        "selected_candidates": list(selected),
        "quarantined_candidates": [
            str(row["feature_name"]) for row in rows if not bool(row["passed"])
        ],
        "top_by_stability": ranked["top_by_stability"],
        "top_by_null_margin": ranked["top_by_null_margin"],
        "near_misses": ranked["near_misses"],
        "candidates": rows,
    }


def render_text_candidate_screen_report(screen: Mapping[str, object]) -> str:
    """Render a compact operator-facing Markdown report for a screen payload."""
    return render_candidate_screen_report(
        screen,
        title="Text Candidate Screen",
        feature_set_key="text_feature_set_version",
        feature_set_label="Text feature set",
        next_action=(
            "Do not register or backfill a new catalyst feature set unless at least "
            "three candidates pass this screen without threshold relaxation."
        ),
    )
