"""Research-campaign diagnostic candidate-screen CLI handlers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quant_platform.research.campaign.screen_support import (
    filter_samples_by_window as _filter_samples_by_window,
)
from quant_platform.research.campaign.screen_support import (
    load_json_mapping as _load_json_mapping,
)
from quant_platform.research.campaign.screen_support import (
    load_sample_build_summary as _load_sample_build_summary,
)
from quant_platform.research.campaign.screen_support import (
    load_text_feature_vectors as _load_text_feature_vectors,
)
from quant_platform.research.campaign.screen_support import (
    sample_filter_payload as _sample_filter_payload,
)
from quant_platform.research.campaign.screen_support import (
    screen_inputs as _screen_inputs,
)
from quant_platform.research.campaign.screen_support import (
    screen_slug as _screen_slug,
)
from quant_platform.research.campaign.screen_support import (
    write_screen_payload as _write_screen_payload,
)
from quant_platform.research.common import (
    _require_durable_research_inputs,
    _verify_postgres_schema_if_configured,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def _research_campaign_screen_text_candidates(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.research_service.sampling.factory import (
        load_supervised_samples,
        walk_forward_object_root,
    )
    from quant_platform.services.research_service.text.candidates.screening import (
        TextCandidateScreenThresholds,
        build_text_candidate_screen,
        render_text_candidate_screen_report,
        text_candidates_for_set,
        write_text_candidate_family_artifacts,
    )

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    output_root: Path = args.output_root or walk_forward_object_root(
        settings.storage.object_store_root,
    )
    all_samples = tuple(load_supervised_samples(args.samples_file))
    samples = _filter_samples_by_window(
        all_samples,
        sample_start=getattr(args, "sample_start", None),
        sample_end=getattr(args, "sample_end", None),
    )
    sample_build = _load_sample_build_summary(args.sample_build_summary)
    if samples != all_samples:
        sample_build = {**sample_build, "samples": len(samples)}
    source_manifest = _load_json_mapping(args.source_data_manifest)
    text_vectors = await _load_text_feature_vectors(
        settings,
        feature_set_version=str(args.text_feature_set_version),
        end=max((sample.as_of for sample in samples), default=datetime.now(tz=UTC)),
    )
    thresholds = TextCandidateScreenThresholds(
        min_source_density=float(args.min_source_density),
        min_null_margin=float(args.min_null_margin),
        min_ic_mean=float(args.min_ic_mean),
        min_icir=float(args.min_icir),
        max_negative_ic_streak=int(args.max_negative_ic_streak),
        min_passing_candidates=int(args.min_passing_candidates),
    )
    candidate_set = str(args.candidate_set)
    screen = build_text_candidate_screen(
        samples=samples,
        text_vectors=text_vectors,
        source_manifest=source_manifest,
        sample_build=sample_build,
        text_feature_set_version=str(args.text_feature_set_version),
        candidate_family=str(args.candidate_family),
        lookback_days=int(args.lookback_days),
        thresholds=thresholds,
        seed=int(args.permutation_seed),
        permutation_count=int(args.permutation_count),
        candidate_set=candidate_set,
        candidates=text_candidates_for_set(candidate_set),
        promoted_feature_set_version=str(args.promoted_feature_set_version),
    )
    screen.update(_screen_inputs(args))
    screen["sample_filter"] = _sample_filter_payload(all_samples, samples, args)
    slug = _screen_slug(args, samples=samples, feature_attr="text_feature_set_version")
    diagnostics_root = output_root / "diagnostics" / slug
    family_artifacts: dict[str, object] = {"written": False}
    if bool(screen["passed"]):
        feature_set = str(args.promoted_feature_set_version)
        family_artifacts = write_text_candidate_family_artifacts(
            screen=screen,
            feature_card_dir=diagnostics_root / "feature_cards" / feature_set,
            feature_family_file=diagnostics_root / "feature_families" / f"{feature_set}.json",
        )
    screen["feature_family_artifacts"] = family_artifacts
    screen["promotion_artifacts_written"] = bool(family_artifacts.get("written", False))
    return _write_screen_payload(
        screen=screen,
        output_root=output_root,
        slug=slug,
        report=render_text_candidate_screen_report(screen),
    )


async def _research_campaign_screen_event_candidates(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.research_service.events.candidates.screening import (
        EventCandidateScreenThresholds,
        build_event_candidate_screen,
        event_candidates_for_set,
        render_event_candidate_screen_report,
        write_event_candidate_family_artifacts,
    )
    from quant_platform.services.research_service.sampling.factory import (
        load_supervised_samples,
        walk_forward_object_root,
    )

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    output_root: Path = args.output_root or walk_forward_object_root(
        settings.storage.object_store_root,
    )
    all_samples = tuple(load_supervised_samples(args.samples_file))
    samples = _filter_samples_by_window(
        all_samples,
        sample_start=getattr(args, "sample_start", None),
        sample_end=getattr(args, "sample_end", None),
    )
    sample_build = _load_sample_build_summary(args.sample_build_summary)
    if samples != all_samples:
        sample_build = {**sample_build, "samples": len(samples)}
    thresholds = EventCandidateScreenThresholds(
        min_source_density=float(args.min_source_density),
        min_null_margin=float(args.min_null_margin),
        min_ic_mean=float(args.min_ic_mean),
        min_icir=float(args.min_icir),
        max_negative_ic_streak=int(args.max_negative_ic_streak),
        min_passing_candidates=int(args.min_passing_candidates),
    )
    candidate_set = str(args.candidate_set)
    screen = build_event_candidate_screen(
        samples=samples,
        source_manifest=_load_json_mapping(args.source_data_manifest),
        sample_build=sample_build,
        event_feature_set_version=str(args.event_feature_set_version),
        candidate_family=str(args.candidate_family),
        thresholds=thresholds,
        seed=int(args.permutation_seed),
        permutation_count=int(args.permutation_count),
        candidate_set=candidate_set,
        candidates=event_candidates_for_set(candidate_set),
    )
    screen.update(_screen_inputs(args))
    screen["sample_filter"] = _sample_filter_payload(all_samples, samples, args)
    slug = _screen_slug(args, samples=samples, feature_attr="event_feature_set_version")
    diagnostics_root = output_root / "diagnostics" / slug
    family_artifacts: dict[str, object] = {"written": False}
    if bool(screen["passed"]):
        feature_set = str(args.event_feature_set_version)
        family_artifacts = write_event_candidate_family_artifacts(
            screen=screen,
            feature_card_dir=diagnostics_root / "feature_cards" / feature_set,
            feature_family_file=diagnostics_root / "feature_families" / f"{feature_set}.json",
        )
    screen["feature_family_artifacts"] = family_artifacts
    screen["promotion_artifacts_written"] = bool(family_artifacts.get("written", False))
    return _write_screen_payload(
        screen=screen,
        output_root=output_root,
        slug=slug,
        report=render_event_candidate_screen_report(screen),
    )
