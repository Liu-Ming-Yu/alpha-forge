"""Intraday candidate-screen CLI handler."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.screen_support import (
    filter_samples_by_window as _filter_samples_by_window,
)
from quant_platform.research.campaign.screen_support import (
    load_sample_build_summary as _load_sample_build_summary,
)
from quant_platform.research.campaign.screen_support import (
    sample_filter_payload as _sample_filter_payload,
)
from quant_platform.research.campaign.screen_support import (
    screen_slug as _screen_slug,
)
from quant_platform.research.campaign.screen_support import (
    write_screen_payload as _write_screen_payload,
)
from quant_platform.research.common import (
    _instrument_lookup_from_contracts,
    _load_instrument_contracts,
)
from quant_platform.services.data_service.intraday.intraday_file_loader import (
    load_vendor_bar_batch_from_file,
)
from quant_platform.services.data_service.intraday.intraday_validation import (
    validate_vendor_bar_batch,
    validation_payload,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.market_data import MarketBar


async def _research_campaign_screen_intraday_candidates(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.research_service.intraday.candidates.screening import (
        IntradayCandidateScreenThresholds,
        build_intraday_candidate_screen,
        intraday_candidates_for_set,
        render_intraday_candidate_screen_report,
        write_intraday_candidate_family_artifacts,
    )
    from quant_platform.services.research_service.sampling.factory import (
        load_supervised_samples,
        walk_forward_object_root,
    )

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
    contracts = _load_instrument_contracts(str(args.contracts_file))
    bars, input_payload, validation_blockers = _load_intraday_screen_bars(
        raw_files=tuple(str(value) for value in args.intraday_file),
        contracts=contracts,
        fetched_at=max((sample.as_of for sample in samples), default=datetime.now(tz=UTC)),
    )
    thresholds = IntradayCandidateScreenThresholds(
        min_source_density=float(args.min_source_density),
        min_null_margin=float(args.min_null_margin),
        min_ic_mean=float(args.min_ic_mean),
        min_icir=float(args.min_icir),
        max_negative_ic_streak=int(args.max_negative_ic_streak),
        min_passing_candidates=int(args.min_passing_candidates),
    )
    candidate_set = str(args.candidate_set)
    screen = build_intraday_candidate_screen(
        samples=samples,
        intraday_bars=bars,
        sample_build=sample_build,
        intraday_feature_set_version=str(args.intraday_feature_set_version),
        candidate_family=str(args.candidate_family),
        thresholds=thresholds,
        seed=int(args.permutation_seed),
        permutation_count=int(args.permutation_count),
        candidate_set=candidate_set,
        candidates=intraday_candidates_for_set(candidate_set),
    )
    screen.update(
        {
            "samples_file": str(args.samples_file),
            "sample_build_summary": str(args.sample_build_summary),
            "contracts_file": str(args.contracts_file),
            "intraday_files": input_payload,
            "sample_filter": _sample_filter_payload(all_samples, samples, args),
        }
    )
    if validation_blockers:
        screen["passed"] = False
        screen["reason"] = "intraday candidate screen blocked prospective feature family"
        screen["blockers"] = [
            *list(cast("Sequence[object]", screen.get("blockers", ()))),
            *validation_blockers,
        ]
    slug = _screen_slug(args, samples=samples, feature_attr="intraday_feature_set_version")
    diagnostics_root = output_root / "diagnostics" / slug
    family_artifacts: dict[str, object] = {"written": False}
    if bool(screen["passed"]):
        feature_set = str(args.intraday_feature_set_version)
        family_artifacts = write_intraday_candidate_family_artifacts(
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
        report=render_intraday_candidate_screen_report(screen),
    )


def _load_intraday_screen_bars(
    *,
    raw_files: tuple[str, ...],
    contracts: dict[uuid.UUID, dict[str, object]],
    fetched_at: datetime,
) -> tuple[tuple[MarketBar, ...], list[dict[str, object]], list[str]]:
    lookup = _instrument_lookup_from_contracts(contracts)
    by_bar_id: dict[uuid.UUID, MarketBar] = {}
    inputs: list[dict[str, object]] = []
    blockers: list[str] = []
    for raw in raw_files:
        vendor, path = _parse_intraday_file(raw)
        batch = load_vendor_bar_batch_from_file(
            path,
            vendor=vendor,
            instrument_lookup=lookup,
            as_of=fetched_at,
        )
        report = validate_vendor_bar_batch(batch)
        payload = validation_payload(report)
        payload["path"] = str(path)
        inputs.append(payload)
        if not report.passed:
            blockers.extend(
                f"{vendor} {issue.code}: {issue.detail}"
                for issue in report.issues
                if issue.severity == "error"
            )
        for bar in batch.bars:
            by_bar_id.setdefault(bar.bar_id, bar)
    bars = tuple(
        sorted(by_bar_id.values(), key=lambda bar: (str(bar.instrument_id), bar.timestamp))
    )
    return bars, inputs, blockers


def _parse_intraday_file(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise OperatorUsageError("--intraday-file must use vendor=/path/to/file")
    vendor, path = raw.split("=", 1)
    if not vendor.strip() or not path.strip():
        raise OperatorUsageError("--intraday-file must use vendor=/path/to/file")
    return vendor.strip(), Path(path)


__all__ = ["_filter_samples_by_window", "_research_campaign_screen_intraday_candidates"]
