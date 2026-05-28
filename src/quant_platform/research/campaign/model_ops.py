"""Model artifact helpers for governed research campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant_platform.services.research_service.campaigns.portfolio.construction import (
    CampaignPortfolioConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def build_campaign_model_artifacts(
    *,
    args: Any,
    campaign_samples: Any,
    admission: Any,
    effective_slippage: float,
    output_root: Path,
    artifact_store: Any,
    campaign_context: dict[str, object],
    feature_audits: Any = (),
    paper_source_weights: Mapping[str, float] | None = None,
) -> tuple[Any, Path, Path | None]:
    from quant_platform.services.research_service.campaigns.metrics.model_comparison import (
        build_linear_model_comparison,
        xgboost_model_comparison_row,
    )
    from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
        WalkForwardConfig,
    )
    from quant_platform.services.research_service.sampling.factory import (
        AlphaEligibilityThresholds,
        write_model_comparison,
        write_walk_forward_artifacts,
    )

    walk_config = WalkForwardConfig(
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_folds=args.min_folds,
    )
    thresholds = AlphaEligibilityThresholds(
        min_oos_rolling_ic=args.min_oos_rolling_ic,
        min_ic_60d=args.min_ic_60d,
        max_fold_negative_ic_streak=args.max_fold_negative_ic_streak,
        max_drawdown=args.max_drawdown,
        min_slippage_adjusted_sharpe=args.min_slippage_adjusted_sharpe,
    )
    model_features = _resolve_model_features(args, admission)
    return_scale = _governed_campaign_return_scale(args, paper_source_weights)
    portfolio_config = _campaign_portfolio_config(args)
    campaign_context["model_features"] = list(model_features)
    campaign_context["return_scale"] = return_scale
    campaign_context["portfolio_config"] = portfolio_config.to_payload()
    evidence, model_comparison_rows = build_linear_model_comparison(
        samples=campaign_samples,
        config=walk_config,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        thresholds=thresholds,
        slippage_bps_per_turnover=effective_slippage,
        feature_names=model_features,
        return_scale=return_scale,
        portfolio_config=portfolio_config,
    )
    evidence = write_walk_forward_artifacts(
        evidence,
        output_root=output_root,
        artifact_store=artifact_store,
    )
    xgboost_manifest_path, xgboost_manifest, xgboost_search_rows = _maybe_train_xgboost(
        args=args,
        campaign_samples=campaign_samples,
        model_features=model_features,
        feature_versions=_feature_versions_from_audits(feature_audits, model_features),
        output_root=output_root,
    )
    model_comparison_rows.append(
        xgboost_model_comparison_row(
            manifest_path=xgboost_manifest_path,
            manifest=xgboost_manifest,
            search_candidates=xgboost_search_rows,
        )
    )
    model_comparison_path = write_model_comparison(
        evidence,
        rows=model_comparison_rows,
        campaign_context=campaign_context,
        artifact_store=artifact_store,
    )
    return evidence, model_comparison_path, xgboost_manifest_path


def _maybe_train_xgboost(
    *,
    args: Any,
    campaign_samples: Any,
    model_features: Any,
    feature_versions: Mapping[str, str],
    output_root: Path,
) -> tuple[Path | None, object | None, list[dict[str, object]]]:
    if not args.train_xgboost:
        return None, None, []
    from quant_platform.research.campaign.xgboost import (
        train_admitted_xgboost_ranker,
    )

    xgboost_manifest, xgboost_manifest_path, xgboost_search_rows = train_admitted_xgboost_ranker(
        samples=campaign_samples,
        admitted_features=model_features,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        feature_versions=feature_versions,
        output_root=output_root,
        device=args.xgboost_device,
        require_gpu=args.xgboost_require_gpu,
        purge_days=args.purge_days,
        search_mode=getattr(args, "xgboost_search", "off"),
    )
    return xgboost_manifest_path, xgboost_manifest, xgboost_search_rows


def _resolve_model_features(args: Any, admission: Any) -> tuple[str, ...]:
    selected = tuple(str(name) for name in (getattr(args, "model_feature", None) or ()))
    admitted = tuple(str(name) for name in admission.admitted_features)
    min_admitted_features = int(getattr(args, "min_admitted_features", 1))
    if len(admitted) < min_admitted_features:
        raise ValueError("--admitted feature count must meet --min-admitted-features")
    if not selected:
        return admitted
    admitted_set = set(admitted)
    missing = tuple(name for name in selected if name not in admitted_set)
    if missing:
        missing_text = ", ".join(repr(name) for name in missing)
        raise ValueError(f"--model-feature must be admitted by feature audit: {missing_text}")
    return selected


def _governed_campaign_return_scale(
    args: Any,
    paper_source_weights: Mapping[str, float] | None,  # noqa: ARG001 - call stability
) -> float:
    """Fail closed when governed campaign eligibility tries to scale P&L."""
    configured = float(getattr(args, "return_scale", 1.0) or 1.0)
    if configured != 1.0:
        raise ValueError("governed campaign eligibility requires return_scale=1.0")
    return 1.0


def _campaign_portfolio_config(args: Any) -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode=str(getattr(args, "campaign_portfolio_mode", "runtime-long-only")),
        top_n=int(getattr(args, "campaign_top_n", 10)),
        vol_target=float(getattr(args, "campaign_vol_target", 0.15)),
        vol_floor=float(getattr(args, "campaign_vol_floor", 0.05)),
        vol_lookback_days=int(getattr(args, "campaign_vol_lookback_days", 63)),
        max_gross_exposure=float(getattr(args, "campaign_max_gross_exposure", 0.60)),
        min_cash_buffer=float(getattr(args, "campaign_min_cash_buffer", 0.05)),
        max_single_name_weight=float(getattr(args, "campaign_max_single_name_weight", 0.05)),
        max_daily_turnover=float(getattr(args, "campaign_max_daily_turnover", 0.20)),
        max_position_change=float(getattr(args, "campaign_max_position_change", 0.05)),
        no_trade_band=float(getattr(args, "campaign_no_trade_band", 0.0)),
        rebalance_interval_days=int(getattr(args, "campaign_rebalance_interval_days", 1)),
    )


def _feature_versions_from_audits(
    feature_audits: Any,
    feature_names: Any,
) -> dict[str, str]:
    wanted = {str(name) for name in feature_names}
    versions: dict[str, str] = {}
    for row in feature_audits or ():
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature_name") or "")
        version = str(row.get("feature_version") or "")
        if name in wanted and version:
            versions[name] = version
    return versions


__all__ = ["build_campaign_model_artifacts"]
