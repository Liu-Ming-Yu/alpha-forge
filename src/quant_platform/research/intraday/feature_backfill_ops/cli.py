"""CLI handler for promoted intraday-alpha feature-vector backfills."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from quant_platform.application.errors import OperatorUsageError
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.research.campaign.intraday_screen import (
    _load_intraday_screen_bars,
)
from quant_platform.research.common import (
    _load_instrument_contracts,
    _require_durable_research_inputs,
    _verify_postgres_schema_if_configured,
    research_json_result,
)
from quant_platform.research.intraday.feature_backfill_ops.samples import (
    _sample_free_intraday_samples,
)
from quant_platform.services.research_service.intraday.candidates.screening import (
    intraday_candidates_for_set,
)
from quant_platform.services.research_service.intraday.features.backfill import (
    backfill_intraday_feature_vectors,
    feature_names_from_family_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_platform.application.research import FeaturesBackfillIntradayAlphaRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.services.research_service.intraday.candidates.features import (
        IntradayCandidateFeatureSpec,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


async def _features_backfill_intraday_alpha(
    settings: PlatformSettings, args: FeaturesBackfillIntradayAlphaRequest
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.research_service.sampling.factory import load_supervised_samples

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    contracts = _load_instrument_contracts(str(args.contracts_file))
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
    )
    samples, sample_input_payload = await _load_backfill_samples(
        feature_repo=session.feature_repo,
        contracts=contracts,
        args=args,
        load_supervised_samples=load_supervised_samples,
    )
    bars, input_payload, blockers = _load_intraday_screen_bars(
        raw_files=tuple(str(value) for value in args.intraday_file),
        contracts=contracts,
        fetched_at=max((sample.as_of for sample in samples), default=datetime.now(tz=UTC)),
    )
    if blockers:
        return research_json_result(
            {
                "passed": False,
                "reason": "intraday alpha feature backfill blocked input validation",
                "blockers": blockers,
                "intraday_files": input_payload,
            },
            passed=False,
        )
    feature_names = feature_names_from_family_file(args.feature_family_file)
    result = await backfill_intraday_feature_vectors(
        samples=samples,
        intraday_bars=bars,
        candidates=cast(
            "Sequence[IntradayCandidateFeatureSpec]",
            intraday_candidates_for_set(str(args.candidate_set)),
        ),
        feature_names=feature_names,
        repo=session.feature_repo,
        strategy_run_id=uuid.uuid4(),
        feature_set_version=str(args.feature_set_version),
        artifact_uri=str(args.artifact_uri or args.feature_family_file.resolve().as_uri()),
        dry_run=bool(args.dry_run),
    )
    completed = result.vector_count > 0 or result.skipped_existing_vectors > 0
    payload = {
        "passed": completed,
        "reason": "intraday alpha feature backfill complete"
        if completed
        else "intraday alpha feature backfill stored no vectors",
        "sample_input": sample_input_payload,
        "contracts_file": str(args.contracts_file),
        "feature_family_file": str(args.feature_family_file),
        "intraday_files": input_payload,
        **result.to_payload(),
    }
    return research_json_result(payload, passed=completed)


async def _load_backfill_samples(
    *,
    feature_repo: FeatureRepository,
    contracts: Mapping[uuid.UUID, Mapping[str, object]],
    args: Any,
    load_supervised_samples: Any,
) -> tuple[tuple[SupervisedAlphaSample, ...], dict[str, object]]:
    samples_file = getattr(args, "samples_file", None)
    if samples_file is not None:
        if getattr(args, "start", None) is not None or getattr(args, "end", None) is not None:
            raise OperatorUsageError("--samples-file is mutually exclusive with --start/--end")
        return tuple(load_supervised_samples(samples_file)), {
            "mode": "samples_file",
            "samples_file": str(samples_file),
        }
    if getattr(args, "start", None) is None or getattr(args, "end", None) is None:
        raise OperatorUsageError(
            "backfill-intraday-alpha requires either --samples-file or --start/--end"
        )
    return await _sample_free_intraday_samples(
        feature_repo=feature_repo,
        contracts=contracts,
        start=args.start,
        end=args.end,
        date_policy=str(getattr(args, "date_policy", "nyse-sessions")),
        context_feature_set_version=str(getattr(args, "context_feature_set_version", "") or ""),
    )
