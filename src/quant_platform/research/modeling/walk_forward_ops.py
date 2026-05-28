"""Walk-forward research operation wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.common import research_json_result

if TYPE_CHECKING:
    from quant_platform.application.research import WalkForwardRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def _walk_forward(
    settings: PlatformSettings,
    request: WalkForwardRequest,
) -> UseCaseResult[dict[str, object]]:
    """Dispatch purged walk-forward research evaluations."""
    if request.command != "run":
        raise OperatorUsageError(f"unknown walk-forward subcommand: {request.command}")

    from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
        WalkForwardConfig,
    )
    from quant_platform.services.research_service.sampling.factory import (
        AlphaEligibilityThresholds,
        load_supervised_samples,
        run_sample_walk_forward,
        walk_forward_object_root,
        write_campaign_manifest,
        write_walk_forward_artifacts,
    )

    output_root = request.output_root or walk_forward_object_root(
        settings.storage.object_store_root
    )
    evidence = run_sample_walk_forward(
        samples=load_supervised_samples(request.samples),
        config=WalkForwardConfig(
            train_window_days=request.train_window_days,
            test_window_days=request.test_window_days,
            step_days=request.step_days,
            purge_days=request.purge_days,
            embargo_days=request.embargo_days,
            min_folds=request.min_folds,
        ),
        model_version=request.model_version,
        feature_set_version=request.feature_set_version,
        thresholds=AlphaEligibilityThresholds(
            min_oos_rolling_ic=request.min_oos_rolling_ic,
            min_ic_60d=request.min_ic_60d,
            max_fold_negative_ic_streak=request.max_fold_negative_ic_streak,
            max_drawdown=request.max_drawdown,
            min_slippage_adjusted_sharpe=request.min_slippage_adjusted_sharpe,
        ),
        slippage_bps_per_turnover=request.slippage_bps_per_turnover,
    )
    evidence = write_walk_forward_artifacts(evidence, output_root=output_root)
    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=request.samples,
        paper_source_weights={},
    )
    passed = bool(evidence.eligibility["passed"])
    payload = {
        "run_id": str(evidence.run_id),
        "artifact_root": str(evidence.artifact_root),
        "campaign_manifest": str(manifest_path),
        "passed": passed,
        "metrics": dict(evidence.metrics),
        "checks": evidence.eligibility["checks"],
    }
    return research_json_result(payload, passed=passed)
