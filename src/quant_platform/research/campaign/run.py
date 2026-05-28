"""Research-campaign run workflow composition."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.admission import resolve_campaign_admission
from quant_platform.research.campaign.inputs import (
    _observed_slippage_bps,
    _parse_paper_source_weights,
)
from quant_platform.research.campaign.model_ops import (
    build_campaign_model_artifacts,
)
from quant_platform.research.campaign.output import (
    campaign_run_payload,
    maybe_write_text_model_manifest,
)
from quant_platform.research.campaign.policy import governed_feature_audit_blocker
from quant_platform.research.campaign.signal_ops import (
    record_campaign_prediction_evidence,
    record_campaign_signal_gates,
)
from quant_platform.research.campaign.source_density import (
    maybe_block_for_catalyst_source_density,
)
from quant_platform.research.campaign.stability_attribution import (
    maybe_run_stability_attribution_preflight,
)
from quant_platform.research.common import (
    _build_samples_to_path,
    _latest_calibration_artifact,
    _load_calibration_recommendation_bps,
    _require_durable_research_inputs,
    _samples_result_payload,
    _verify_postgres_schema_if_configured,
    research_json_result,
)

if TYPE_CHECKING:
    from quant_platform.application.research import CampaignRunRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

__all__ = ["_research_campaign_run"]


async def _research_campaign_run(
    settings: PlatformSettings, args: CampaignRunRequest
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.infrastructure.support.artifact_store import FileSystemArtifactStore
    from quant_platform.services.research_service.sampling.factory import (
        calibrated_slippage_bps_per_turnover,
        load_supervised_samples,
        walk_forward_object_root,
        write_campaign_manifest,
    )

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    if payload := governed_feature_audit_blocker(args):
        return research_json_result(payload, passed=False)
    signal_type = _campaign_signal_type(args)

    output_root = args.output_root or walk_forward_object_root(settings.storage.object_store_root)
    artifact_store = FileSystemArtifactStore(output_root)
    paper_weights = _parse_paper_source_weights(settings, args.paper_source_weights_json)
    sample_slug = f"{args.model_version}_{args.start:%Y-%m-%d}_{args.end:%Y-%m-%d}"
    sample_path = output_root / "_inputs" / sample_slug / "samples.json"
    sample_path, sample_result = await _build_samples_to_path(
        settings=settings,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        output=sample_path,
        feature_set_version=args.feature_set_version,
        horizon_days=args.horizon_days,
        bar_seconds=args.bar_seconds,
        max_feature_age_days=args.max_feature_age_days,
        date_policy=str(args.date_policy),
    )
    sample_build_payload = _samples_result_payload(sample_path, sample_result)
    sample_build_payload.update(
        {
            "sample_start": args.start.isoformat(),
            "sample_end": args.end.isoformat(),
            "contracts_file": str(args.contracts_file),
            "universe": {"contracts_file": str(args.contracts_file)},
            "feature_set_version": str(args.feature_set_version),
            "horizon_days": int(args.horizon_days),
            "bar_seconds": int(args.bar_seconds),
            "max_feature_age_days": int(args.max_feature_age_days),
        }
    )

    observed_slippage = await _observed_slippage_bps(settings)
    calibration_path = _latest_calibration_artifact(settings)
    calibration_bps, calibration_meta = _load_calibration_recommendation_bps(
        calibration_path,
        max_age_days=float(getattr(args, "max_calibration_age_days", 14.0) or 14.0),
        as_of=datetime.now(tz=UTC),
    )
    if bool(getattr(args, "require_calibration", False)) and calibration_bps is None:
        return research_json_result(
            {
                "passed": False,
                "reason": "fresh simulator calibration artifact is required",
                "calibration_artifact": calibration_meta,
            },
            passed=False,
        )
    effective_slippage = calibrated_slippage_bps_per_turnover(
        observed_slippage,
        default_bps=float(args.slippage_bps_per_turnover),
        calibration_recommendation_bps=calibration_bps,
    )
    if observed_slippage is not None or calibration_bps is not None:
        log.info(
            "research_campaign.slippage_calibrated",
            observed_slippage_bps=observed_slippage,
            configured_slippage_bps=float(args.slippage_bps_per_turnover),
            effective_slippage_bps=effective_slippage,
            calibration_recommendation_bps=calibration_bps,
            calibration_artifact=calibration_meta,
        )

    campaign_samples = load_supervised_samples(sample_path)
    source_density_block = maybe_block_for_catalyst_source_density(
        feature_set_version=str(args.feature_set_version),
        source_data_manifest=getattr(args, "source_data_manifest", None),
        samples=campaign_samples,
        output_root=output_root,
        sample_slug=sample_slug,
    )
    if source_density_block is not None:
        return research_json_result(source_density_block, passed=False)
    preflight = await maybe_run_stability_attribution_preflight(
        settings=settings,
        args=args,
        samples=campaign_samples,
        sample_build=sample_build_payload,
        output_root=output_root,
        sample_slug=sample_slug,
        slippage_bps_per_turnover=effective_slippage,
    )
    if preflight is not None and not preflight.passed:
        return research_json_result(preflight.blocked_payload or {}, passed=False)
    feature_diagnostics_path = getattr(args, "feature_diagnostics", None)
    feature_attribution_path = None
    if preflight is not None:
        feature_diagnostics_path = preflight.direction_path
        feature_attribution_path = preflight.attribution_path
    admission_resolution = await resolve_campaign_admission(
        settings=settings,
        samples=campaign_samples,
        sample_build=sample_build_payload,
        output_root=output_root,
        sample_slug=sample_slug,
        feature_set_version=args.feature_set_version,
        horizon_days=args.horizon_days,
        slippage_bps_per_turnover=effective_slippage,
        feature_audit_mode=str(args.feature_audit_mode),
        feature_card_dir=args.feature_card_dir,
        feature_admission=args.feature_admission,
        min_admitted_features=int(args.min_admitted_features),
        date_policy=str(args.date_policy),
        feature_diagnostics_path=feature_diagnostics_path,
        feature_attribution_path=feature_attribution_path,
    )
    if not admission_resolution.admission.passed:
        return research_json_result(admission_resolution.blocked_payload or {}, passed=False)

    evidence, model_comparison_path, xgboost_manifest_path = build_campaign_model_artifacts(
        args=args,
        campaign_samples=campaign_samples,
        admission=admission_resolution.admission,
        effective_slippage=effective_slippage,
        output_root=output_root,
        artifact_store=artifact_store,
        campaign_context=admission_resolution.campaign_context,
        feature_audits=admission_resolution.feature_audits,
        paper_source_weights=paper_weights,
    )

    signal_status, model_signal_status = await record_campaign_signal_gates(
        settings,
        model_version=args.model_version,
        train_xgboost=bool(args.train_xgboost),
        signal_type=signal_type,
        as_of=args.end,
        daily_ic=float(evidence.metrics["oos_rolling_ic"]),
        observations=int(evidence.metrics["daily_observations"]),
        drawdown=0.0,
        turnover=1.0,
        source_weights=paper_weights,
        daily_ics=evidence.daily_ics,
    )
    prediction_evidence_counts = await record_campaign_prediction_evidence(
        settings,
        samples=campaign_samples,
        source_weights=paper_weights,
        signal_type=signal_type,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        as_of=args.end,
        selected_weights=evidence.selected_weights,
    )
    admission_resolution.campaign_context["prediction_evidence"] = {
        "counts": dict(prediction_evidence_counts),
        "source_weights": dict(paper_weights),
    }
    manifest_path = write_campaign_manifest(
        evidence,
        samples_path=sample_path,
        paper_source_weights=paper_weights,
        xgboost_manifest_path=xgboost_manifest_path,
        model_comparison_path=model_comparison_path,
        feature_audits=admission_resolution.feature_audits,
        campaign_context=admission_resolution.campaign_context,
        command="research-campaign run",
        artifact_store=artifact_store,
    )
    text_manifest_path = maybe_write_text_model_manifest(
        settings=settings,
        args=args,
        signal_type=signal_type,
        output_root=output_root,
        admission=admission_resolution.admission,
        evidence=evidence,
        campaign_manifest=manifest_path,
    )
    payload = campaign_run_payload(
        evidence=evidence,
        manifest_path=manifest_path,
        sample_build=sample_build_payload,
        paper_weights=paper_weights,
        feature_audits=admission_resolution.feature_audits,
        admission=admission_resolution.admission,
        model_comparison_path=model_comparison_path,
        text_manifest_path=text_manifest_path,
        xgboost_manifest_path=xgboost_manifest_path,
        signal_status=signal_status,
        model_signal_status=model_signal_status,
    )
    payload["prediction_evidence_counts"] = prediction_evidence_counts
    passed = not (args.fail_on_ineligible and not evidence.eligibility["passed"])
    return research_json_result(payload, passed=passed)


def _campaign_signal_type(args: CampaignRunRequest) -> str:
    raw = str(getattr(args, "signal_type", "auto") or "auto")
    if raw == "auto":
        return "xgboost" if bool(getattr(args, "train_xgboost", False)) else "classical"
    if raw == "xgboost" and not bool(getattr(args, "train_xgboost", False)):
        raise OperatorUsageError("--signal-type xgboost requires --train-xgboost")
    if raw in {"text", "event", "intraday"} and bool(getattr(args, "train_xgboost", False)):
        raise OperatorUsageError(f"--signal-type {raw} cannot be combined with --train-xgboost")
    return raw
