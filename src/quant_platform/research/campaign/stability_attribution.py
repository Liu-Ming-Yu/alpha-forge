"""Stability-attribution routing for governed research campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.stability_preflight import (
    StabilityAttributionPreflight,
    run_stability_attribution_preflight,
)
from quant_platform.services.research_service.feature_quality.diagnostics import (
    null_qualified_features,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.config import PlatformSettings
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


async def maybe_run_stability_attribution_preflight(
    *,
    settings: PlatformSettings,
    args: Any,
    samples: Sequence[SupervisedAlphaSample],
    sample_build: Mapping[str, object],
    output_root: Path,
    sample_slug: str,
    slippage_bps_per_turnover: float,
) -> StabilityAttributionPreflight | None:
    """Run stability attribution only for current governed campaigns."""
    feature_set_version = str(args.feature_set_version)
    governed = _governed_family_files()
    if feature_set_version not in governed:
        return None
    if int(args.horizon_days) != 21:
        raise OperatorUsageError(f"{feature_set_version} campaigns must keep --horizon-days 21")
    family_file = args.feature_family_file or Path(governed[feature_set_version])
    return await run_stability_attribution_preflight(
        settings=settings,
        samples=samples,
        sample_build=sample_build,
        output_root=output_root,
        sample_slug=sample_slug,
        contracts_file=str(args.contracts_file),
        start=args.start,
        end=args.end,
        feature_set_version=feature_set_version,
        official_horizon_days=int(args.horizon_days),
        horizons=tuple(int(value) for value in args.attribution_horizons),
        bar_seconds=int(args.bar_seconds),
        max_feature_age_days=int(args.max_feature_age_days),
        date_policy=str(args.date_policy),
        feature_card_dir=args.feature_card_dir,
        feature_family_file=family_file,
        slippage_bps_per_turnover=slippage_bps_per_turnover,
        permutation_seed=int(args.attribution_permutation_seed),
        permutation_count=int(args.attribution_permutation_count),
        correlation_threshold=float(args.attribution_correlation_threshold),
        min_null_qualified_features=int(args.min_null_qualified_features),
        candidate_feature_names=_candidate_feature_names(
            feature_set_version=feature_set_version,
            feature_card_dir=args.feature_card_dir,
        ),
    )


def _governed_family_files() -> dict[str, str]:
    from quant_platform.services.research_service.features.paper_alpha.composite import (
        PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
        PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION,
    )
    from quant_platform.services.research_service.features.paper_alpha.event import (
        PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
    )
    from quant_platform.services.research_service.features.paper_alpha.text_features import (
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    )

    return {
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION: (
            "infra/config/feature_families/paper-alpha-catalyst-v10.json"
        ),
        PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION: (
            "infra/config/feature_families/paper-alpha-event-reaction-v2.json"
        ),
        PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION: (
            "infra/config/feature_families/paper-alpha-intraday-microstructure-v2.json"
        ),
        PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION: (
            "infra/config/feature_families/paper-alpha-composite-v1.json"
        ),
    }


def _candidate_feature_names(
    *,
    feature_set_version: str,
    feature_card_dir: Path | None,
) -> tuple[str, ...] | None:
    from quant_platform.services.research_service.features.paper_alpha.text_features import (
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    )

    if (
        feature_set_version != PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION
        or feature_card_dir is None
    ):
        return None
    return tuple(sorted(path.stem for path in feature_card_dir.glob("*.json")))


__all__ = [
    "StabilityAttributionPreflight",
    "maybe_run_stability_attribution_preflight",
    "null_qualified_features",
    "run_stability_attribution_preflight",
]
