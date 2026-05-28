"""Argparse-namespace to typed research request DTO factories.

Each factory converts an :class:`argparse.Namespace` into one of the frozen
research request DTOs. The ``command`` discriminator is read from the relevant
argparse subcommand destination; optional fields use ``getattr`` so a single
factory can serve every leaf parser in its command family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.research import (
    AlphaRequest,
    BacktestEvidenceAssertRequest,
    BacktestIntradayRequest,
    BacktestRunRequest,
    BoostingRequest,
    CampaignAttributeFailuresRequest,
    CampaignDiagnoseFeaturesRequest,
    CampaignPromoteRequest,
    CampaignRunRequest,
    CampaignScreenRequest,
    FeatureAuditRequest,
    FeaturesBackfillIntradayAlphaRequest,
    FeaturesBackfillRequest,
    FeaturesBuildSamplesRequest,
    FeaturesRetentionRequest,
    ModelRegistryRequest,
    WalkForwardRequest,
)

if TYPE_CHECKING:
    import argparse


def model_registry_request(args: argparse.Namespace) -> ModelRegistryRequest:
    return ModelRegistryRequest(
        command=args.mr_command,
        name=getattr(args, "name", ""),
        version=getattr(args, "version", ""),
        engine_version=getattr(args, "engine_version", ""),
        feature_set_version=getattr(args, "feature_set_version", ""),
        config_path=getattr(args, "config_path", None),
        metadata_path=getattr(args, "metadata_path", None),
        artifact_manifest=getattr(args, "artifact_manifest", None),
        from_version=getattr(args, "from_version", ""),
        to_version=getattr(args, "to_version", ""),
    )


def boosting_request(args: argparse.Namespace) -> BoostingRequest:
    from pathlib import Path

    return BoostingRequest(
        command=args.boosting_command,
        samples=getattr(args, "samples", None),
        model_version=getattr(args, "model_version", ""),
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        output_root=getattr(args, "output_root", Path("data/models/xgboost")),
        device=getattr(args, "device", "auto"),
        require_gpu=getattr(args, "require_gpu", False),
        validation_fraction=getattr(args, "validation_fraction", 0.20),
        purge_days=getattr(args, "purge_days", 21),
        num_boost_round=getattr(args, "num_boost_round", 100),
        early_stopping_rounds=getattr(args, "early_stopping_rounds", 10),
        max_depth=getattr(args, "max_depth", 4),
        eta=getattr(args, "eta", 0.05),
        subsample=getattr(args, "subsample", 0.80),
        colsample_bytree=getattr(args, "colsample_bytree", 0.80),
        min_child_weight=getattr(args, "min_child_weight", 1.0),
        random_seed=getattr(args, "random_seed", 17),
    )


def alpha_request(args: argparse.Namespace) -> AlphaRequest:
    return AlphaRequest(
        command=args.alpha_command,
        signal_name=getattr(args, "signal_name", ""),
        signal_type=getattr(args, "signal_type", ""),
        as_of=getattr(args, "as_of", None),
        artifact_manifest=getattr(args, "artifact_manifest", None),
        model_version=getattr(args, "model_version", ""),
        feature_set_version=getattr(args, "feature_set_version", ""),
        engine_version=getattr(args, "engine_version", ""),
        rollback_target=getattr(args, "rollback_target", ""),
        target_version=getattr(args, "target_version", ""),
        contracts_file=getattr(args, "contracts_file", None),
        source=tuple(getattr(args, "source", None) or ()),
        horizon=getattr(args, "horizon", "21d"),
        xgboost_manifest=getattr(args, "xgboost_manifest", None),
        fail_on_missing=getattr(args, "fail_on_missing", False),
        clean_live_days=getattr(args, "clean_live_days", 0),
    )


def walk_forward_request(args: argparse.Namespace) -> WalkForwardRequest:
    return WalkForwardRequest(
        command=args.walk_forward_command,
        samples=args.samples,
        model_version=args.model_version,
        output_root=getattr(args, "output_root", None),
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        train_window_days=getattr(args, "train_window_days", 252),
        test_window_days=getattr(args, "test_window_days", 21),
        step_days=getattr(args, "step_days", 21),
        purge_days=getattr(args, "purge_days", 21),
        embargo_days=getattr(args, "embargo_days", 0),
        min_folds=getattr(args, "min_folds", 3),
        min_oos_rolling_ic=getattr(args, "min_oos_rolling_ic", 0.05),
        min_ic_60d=getattr(args, "min_ic_60d", 0.03),
        max_fold_negative_ic_streak=getattr(args, "max_fold_negative_ic_streak", 2),
        max_drawdown=getattr(args, "max_drawdown", -0.20),
        min_slippage_adjusted_sharpe=getattr(args, "min_slippage_adjusted_sharpe", 1.0),
        slippage_bps_per_turnover=getattr(args, "slippage_bps_per_turnover", 10.0),
    )


def features_build_samples_request(args: argparse.Namespace) -> FeaturesBuildSamplesRequest:
    return FeaturesBuildSamplesRequest(
        command=args.features_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        output=args.output,
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        horizon_days=getattr(args, "horizon_days", 21),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        max_feature_age_days=getattr(args, "max_feature_age_days", 3),
    )


def features_backfill_request(args: argparse.Namespace) -> FeaturesBackfillRequest:
    return FeaturesBackfillRequest(
        command=args.features_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        lookback_days=getattr(args, "lookback_days", 380),
        source_data_manifest=getattr(args, "source_data_manifest", None),
        dry_run=getattr(args, "dry_run", False),
    )


def features_backfill_intraday_alpha_request(
    args: argparse.Namespace,
) -> FeaturesBackfillIntradayAlphaRequest:
    return FeaturesBackfillIntradayAlphaRequest(
        command=args.features_command,
        contracts_file=args.contracts_file,
        intraday_file=tuple(getattr(args, "intraday_file", None) or ()),
        feature_family_file=args.feature_family_file,
        samples_file=getattr(args, "samples_file", None),
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
        context_feature_set_version=getattr(args, "context_feature_set_version", ""),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        feature_set_version=getattr(
            args, "feature_set_version", "paper-alpha-intraday-microstructure-v2"
        ),
        candidate_set=getattr(args, "candidate_set", "microstructure-v2"),
        artifact_uri=getattr(args, "artifact_uri", ""),
        dry_run=getattr(args, "dry_run", False),
    )


def features_retention_request(args: argparse.Namespace) -> FeaturesRetentionRequest:
    return FeaturesRetentionRequest(
        command=args.features_command,
        keep_days=args.keep_days,
        dry_run=getattr(args, "dry_run", False),
    )


def feature_audit_request(args: argparse.Namespace) -> FeatureAuditRequest:
    return FeatureAuditRequest(
        command=args.feature_audit_command,
        feature_card=getattr(args, "feature_card", None),
        samples=getattr(args, "samples", None),
        contracts_file=getattr(args, "contracts_file", None),
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        horizon_days=getattr(args, "horizon_days", 21),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        max_feature_age_days=getattr(args, "max_feature_age_days", 3),
        output_root=getattr(args, "output_root", None),
        baseline_features=getattr(args, "baseline_features", ""),
        slippage_bps_per_turnover=getattr(args, "slippage_bps_per_turnover", 10.0),
        min_daily_groups=getattr(args, "min_daily_groups", 252),
        min_coverage=getattr(args, "min_coverage", 0.95),
        min_oos_ic=getattr(args, "min_oos_ic", 0.02),
        min_icir=getattr(args, "min_icir", 0.10),
        max_negative_ic_streak=getattr(args, "max_negative_ic_streak", 3),
        max_turnover=getattr(args, "max_turnover", 4.0),
        persist=getattr(args, "persist", False),
        feature_name=getattr(args, "feature_name", None),
        feature_version=getattr(args, "feature_version", None),
        limit=getattr(args, "limit", 20),
        manifest=getattr(args, "manifest", None),
        minimum_state=getattr(args, "minimum_state", "paper"),
        reason=getattr(args, "reason", "operator retired"),
    )


def backtest_run_request(args: argparse.Namespace) -> BacktestRunRequest:
    from pathlib import Path

    return BacktestRunRequest(
        command=args.backtest_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        initial_capital=getattr(args, "initial_capital", 100_000.0),
        strategy_name=getattr(args, "strategy_name", "vectorbt_backtest"),
        strategy_version=getattr(args, "strategy_version", "0.1.0"),
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        rebalance_every=getattr(args, "rebalance_every", 1),
        top_n=getattr(args, "top_n", 10),
        output_root=getattr(args, "output_root", Path("data/backtest")),
    )


def backtest_intraday_request(args: argparse.Namespace) -> BacktestIntradayRequest:
    from pathlib import Path

    return BacktestIntradayRequest(
        command=args.backtest_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        decision_time=tuple(getattr(args, "decision_time", None) or ()),
        data_file=getattr(args, "data_file", None),
        vendor=getattr(args, "vendor", "file"),
        initial_capital=getattr(args, "initial_capital", 100_000.0),
        strategy_name=getattr(args, "strategy_name", "intraday_backtest"),
        strategy_version=getattr(args, "strategy_version", "0.1.0"),
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        model_version=getattr(args, "model_version", "classical"),
        universe_name=getattr(args, "universe_name", "intraday_research"),
        dataset_id=tuple(getattr(args, "dataset_id", None) or ()),
        output_root=getattr(args, "output_root", Path("data/backtest")),
    )


def backtest_evidence_assert_request(args: argparse.Namespace) -> BacktestEvidenceAssertRequest:
    return BacktestEvidenceAssertRequest(
        command=args.backtest_evidence_command,
        manifest=args.manifest,
    )


def campaign_run_request(args: argparse.Namespace) -> CampaignRunRequest:
    return CampaignRunRequest(
        command=args.research_campaign_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        model_version=args.model_version,
        feature_set_version=getattr(args, "feature_set_version", "1.0.0"),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        horizon_days=getattr(args, "horizon_days", 21),
        signal_type=getattr(args, "signal_type", "auto"),
        output_root=getattr(args, "output_root", None),
        train_xgboost=getattr(args, "train_xgboost", False),
        xgboost_search=getattr(args, "xgboost_search", "off"),
        xgboost_device=getattr(args, "xgboost_device", "auto"),
        xgboost_require_gpu=getattr(args, "xgboost_require_gpu", False),
        paper_source_weights_json=getattr(args, "paper_source_weights_json", ""),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        max_feature_age_days=getattr(args, "max_feature_age_days", 3),
        train_window_days=getattr(args, "train_window_days", 252),
        test_window_days=getattr(args, "test_window_days", 21),
        step_days=getattr(args, "step_days", 21),
        purge_days=getattr(args, "purge_days", 21),
        embargo_days=getattr(args, "embargo_days", 0),
        min_folds=getattr(args, "min_folds", 3),
        min_oos_rolling_ic=getattr(args, "min_oos_rolling_ic", 0.05),
        min_ic_60d=getattr(args, "min_ic_60d", 0.03),
        max_fold_negative_ic_streak=getattr(args, "max_fold_negative_ic_streak", 2),
        max_drawdown=getattr(args, "max_drawdown", -0.20),
        min_slippage_adjusted_sharpe=getattr(args, "min_slippage_adjusted_sharpe", 1.0),
        slippage_bps_per_turnover=getattr(args, "slippage_bps_per_turnover", 10.0),
        return_scale=getattr(args, "return_scale", 1.0),
        campaign_portfolio_mode=getattr(args, "campaign_portfolio_mode", "runtime-long-only"),
        campaign_top_n=getattr(args, "campaign_top_n", 10),
        campaign_vol_target=getattr(args, "campaign_vol_target", 0.15),
        campaign_vol_floor=getattr(args, "campaign_vol_floor", 0.05),
        campaign_vol_lookback_days=getattr(args, "campaign_vol_lookback_days", 63),
        campaign_max_gross_exposure=getattr(args, "campaign_max_gross_exposure", 0.60),
        campaign_min_cash_buffer=getattr(args, "campaign_min_cash_buffer", 0.05),
        campaign_max_single_name_weight=getattr(args, "campaign_max_single_name_weight", 0.05),
        campaign_max_daily_turnover=getattr(args, "campaign_max_daily_turnover", 0.20),
        campaign_max_position_change=getattr(args, "campaign_max_position_change", 0.05),
        campaign_no_trade_band=getattr(args, "campaign_no_trade_band", 0.0),
        campaign_rebalance_interval_days=getattr(args, "campaign_rebalance_interval_days", 1),
        max_calibration_age_days=getattr(args, "max_calibration_age_days", 14.0),
        require_calibration=getattr(args, "require_calibration", False),
        feature_audit_mode=getattr(args, "feature_audit_mode", "shadow"),
        feature_card_dir=getattr(args, "feature_card_dir", None),
        feature_diagnostics=getattr(args, "feature_diagnostics", None),
        feature_family_file=getattr(args, "feature_family_file", None),
        source_data_manifest=getattr(args, "source_data_manifest", None),
        text_prompt_version=getattr(args, "text_prompt_version", ""),
        attribution_horizons=tuple(getattr(args, "attribution_horizons", None) or (5, 10, 21)),
        attribution_permutation_seed=getattr(args, "attribution_permutation_seed", 17),
        attribution_permutation_count=getattr(args, "attribution_permutation_count", 200),
        attribution_correlation_threshold=getattr(args, "attribution_correlation_threshold", 0.70),
        min_null_qualified_features=getattr(args, "min_null_qualified_features", 3),
        feature_admission=getattr(args, "feature_admission", "passing"),
        min_admitted_features=getattr(args, "min_admitted_features", 3),
        model_feature=(tuple(args.model_feature) if getattr(args, "model_feature", None) else None),
        fail_on_ineligible=getattr(args, "fail_on_ineligible", False),
    )


def campaign_diagnose_features_request(
    args: argparse.Namespace,
) -> CampaignDiagnoseFeaturesRequest:
    return CampaignDiagnoseFeaturesRequest(
        command=args.research_campaign_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        feature_card_dir=args.feature_card_dir,
        feature_set_version=getattr(args, "feature_set_version", "paper-alpha-catalyst-v10"),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        output_root=getattr(args, "output_root", None),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        max_feature_age_days=getattr(args, "max_feature_age_days", 3),
        slippage_bps_per_turnover=getattr(args, "slippage_bps_per_turnover", 10.0),
        max_calibration_age_days=getattr(args, "max_calibration_age_days", 14.0),
        require_calibration=getattr(args, "require_calibration", False),
        horizon_days=getattr(args, "horizon_days", 21),
    )


def campaign_attribute_failures_request(
    args: argparse.Namespace,
) -> CampaignAttributeFailuresRequest:
    from pathlib import Path

    return CampaignAttributeFailuresRequest(
        command=args.research_campaign_command,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        feature_card_dir=args.feature_card_dir,
        feature_set_version=getattr(args, "feature_set_version", "paper-alpha-catalyst-v10"),
        date_policy=getattr(args, "date_policy", "nyse-sessions"),
        output_root=getattr(args, "output_root", None),
        bar_seconds=getattr(args, "bar_seconds", 86400),
        max_feature_age_days=getattr(args, "max_feature_age_days", 3),
        slippage_bps_per_turnover=getattr(args, "slippage_bps_per_turnover", 10.0),
        max_calibration_age_days=getattr(args, "max_calibration_age_days", 14.0),
        require_calibration=getattr(args, "require_calibration", False),
        horizons=tuple(getattr(args, "horizons", None) or (5, 10, 21)),
        official_horizon_days=getattr(args, "official_horizon_days", 21),
        feature_family_file=getattr(
            args,
            "feature_family_file",
            Path("infra/config/feature_families/paper-alpha-catalyst-v10.json"),
        ),
        permutation_seed=getattr(args, "permutation_seed", 17),
        permutation_count=getattr(args, "permutation_count", 200),
        correlation_threshold=getattr(args, "correlation_threshold", 0.70),
    )


def campaign_screen_request(args: argparse.Namespace) -> CampaignScreenRequest:
    return CampaignScreenRequest(
        command=args.research_campaign_command,
        samples_file=args.samples_file,
        sample_build_summary=args.sample_build_summary,
        sample_start=getattr(args, "sample_start", None),
        sample_end=getattr(args, "sample_end", None),
        output_root=getattr(args, "output_root", None),
        screen_name=getattr(args, "screen_name", ""),
        source_data_manifest=getattr(args, "source_data_manifest", None),
        contracts_file=getattr(args, "contracts_file", None),
        intraday_file=tuple(getattr(args, "intraday_file", None) or ()),
        text_feature_set_version=getattr(args, "text_feature_set_version", "text-v5"),
        promoted_feature_set_version=getattr(
            args, "promoted_feature_set_version", "paper-alpha-catalyst-v10"
        ),
        event_feature_set_version=getattr(
            args, "event_feature_set_version", "paper-alpha-event-reaction-v2"
        ),
        intraday_feature_set_version=getattr(
            args, "intraday_feature_set_version", "paper-alpha-intraday-microstructure-v2"
        ),
        candidate_family=getattr(args, "candidate_family", ""),
        candidate_set=getattr(args, "candidate_set", "v10-alpha-quality"),
        lookback_days=getattr(args, "lookback_days", 21),
        min_source_density=getattr(args, "min_source_density", 0.05),
        min_null_margin=getattr(args, "min_null_margin", 0.0),
        min_ic_mean=getattr(args, "min_ic_mean", 0.02),
        min_icir=getattr(args, "min_icir", 0.10),
        max_negative_ic_streak=getattr(args, "max_negative_ic_streak", 3),
        min_passing_candidates=getattr(args, "min_passing_candidates", 3),
        permutation_seed=getattr(args, "permutation_seed", 17),
        permutation_count=getattr(args, "permutation_count", 200),
    )


def campaign_promote_request(args: argparse.Namespace) -> CampaignPromoteRequest:
    return CampaignPromoteRequest(
        command=args.research_campaign_command,
        main_screen=args.main_screen,
        confirmation_screen=args.confirmation_screen,
        full_screen=args.full_screen,
        feature_card_dir=args.feature_card_dir,
        feature_family_file=args.feature_family_file,
        min_passing_candidates=getattr(args, "min_passing_candidates", 3),
        output_root=getattr(args, "output_root", None),
        screen_name=getattr(args, "screen_name", ""),
    )


__all__ = [
    "alpha_request",
    "backtest_evidence_assert_request",
    "backtest_intraday_request",
    "backtest_run_request",
    "boosting_request",
    "campaign_attribute_failures_request",
    "campaign_diagnose_features_request",
    "campaign_promote_request",
    "campaign_run_request",
    "campaign_screen_request",
    "feature_audit_request",
    "features_backfill_intraday_alpha_request",
    "features_backfill_request",
    "features_build_samples_request",
    "features_retention_request",
    "model_registry_request",
    "walk_forward_request",
]
