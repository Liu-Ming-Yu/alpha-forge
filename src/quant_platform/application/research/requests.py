"""Typed request DTOs for the research operator command family.

Every research subcommand carries an explicit frozen request. The first field
``command`` is the subcommand discriminator, mirroring the governance request
DTOs (for example ``SignalGateRequest``). Field names match the argparse
destinations so request factories and downstream ops read the same attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

# --------------------------------------------------------------------------
# model-registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRegistryRequest:
    """``research model-registry {list,promote,retire,diff,rollback}``."""

    command: str
    name: str = ""
    version: str = ""
    engine_version: str = ""
    feature_set_version: str = ""
    config_path: Path | None = None
    metadata_path: Path | None = None
    artifact_manifest: Path | None = None
    from_version: str = ""
    to_version: str = ""


# --------------------------------------------------------------------------
# boosting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoostingRequest:
    """``research boosting {gpu-check,train}``."""

    command: str
    samples: Path | None = None
    model_version: str = ""
    feature_set_version: str = "1.0.0"
    output_root: Path = Path("data/models/xgboost")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    require_gpu: bool = False
    validation_fraction: float = 0.20
    purge_days: int = 21
    num_boost_round: int = 100
    early_stopping_rounds: int = 10
    max_depth: int = 4
    eta: float = 0.05
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    min_child_weight: float = 1.0
    random_seed: int = 17


# --------------------------------------------------------------------------
# alpha
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaRequest:
    """``research alpha {assert,promote,rollback,materialize-forecasts,ramp}``."""

    command: str
    signal_name: str = ""
    signal_type: str = ""
    as_of: datetime | None = None
    artifact_manifest: Path | None = None
    model_version: str = ""
    feature_set_version: str = ""
    engine_version: str = ""
    rollback_target: str = ""
    target_version: str = ""
    contracts_file: Path | None = None
    source: tuple[str, ...] = ()
    horizon: str = "21d"
    xgboost_manifest: Path | None = None
    fail_on_missing: bool = False
    clean_live_days: int = 0


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardRequest:
    """``research walk-forward run``."""

    command: str
    samples: Path
    model_version: str
    output_root: Path | None = None
    feature_set_version: str = "1.0.0"
    train_window_days: int = 252
    test_window_days: int = 21
    step_days: int = 21
    purge_days: int = 21
    embargo_days: int = 0
    min_folds: int = 3
    min_oos_rolling_ic: float = 0.05
    min_ic_60d: float = 0.03
    # Fold-level streak — see AlphaEligibilityThresholds for the unit rationale.
    max_fold_negative_ic_streak: int = 2
    max_drawdown: float = -0.20
    min_slippage_adjusted_sharpe: float = 1.0
    slippage_bps_per_turnover: float = 10.0


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturesBuildSamplesRequest:
    """``research features build-samples``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    output: Path
    feature_set_version: str = "1.0.0"
    date_policy: str = "nyse-sessions"
    horizon_days: int = 21
    bar_seconds: int = 86400
    max_feature_age_days: int = 3


@dataclass(frozen=True)
class FeaturesBackfillRequest:
    """``research features backfill``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    feature_set_version: str = "1.0.0"
    date_policy: str = "nyse-sessions"
    bar_seconds: int = 86400
    lookback_days: int = 380
    source_data_manifest: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class FeaturesBackfillIntradayAlphaRequest:
    """``research features backfill-intraday-alpha``."""

    command: str
    contracts_file: Path
    intraday_file: tuple[str, ...]
    feature_family_file: Path
    samples_file: Path | None = None
    start: datetime | None = None
    end: datetime | None = None
    context_feature_set_version: str = ""
    date_policy: str = "nyse-sessions"
    feature_set_version: str = "paper-alpha-intraday-microstructure-v2"
    candidate_set: str = "seed"
    artifact_uri: str = ""
    dry_run: bool = False


@dataclass(frozen=True)
class FeaturesRetentionRequest:
    """``research features retention``."""

    command: str
    keep_days: int
    dry_run: bool = False


@dataclass(frozen=True)
class FeatureAuditRequest:
    """``research features audit {run,status,assert,retire}``."""

    command: str
    feature_card: Path | None = None
    samples: Path | None = None
    contracts_file: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    feature_set_version: str = "1.0.0"
    horizon_days: int = 21
    bar_seconds: int = 86400
    max_feature_age_days: int = 3
    output_root: Path | None = None
    baseline_features: str = ""
    slippage_bps_per_turnover: float = 10.0
    min_daily_groups: int = 252
    min_coverage: float = 0.95
    min_oos_ic: float = 0.02
    min_icir: float = 0.10
    max_negative_ic_streak: int = 3
    max_turnover: float = 4.0
    persist: bool = False
    feature_name: str | None = None
    feature_version: str | None = None
    limit: int = 20
    manifest: Path | None = None
    minimum_state: str = "paper"
    reason: str = "operator retired"


FeaturesRequest = (
    FeaturesBuildSamplesRequest
    | FeaturesBackfillRequest
    | FeaturesBackfillIntradayAlphaRequest
    | FeatureAuditRequest
)


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestRunRequest:
    """``research backtest run``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    initial_capital: float = 100_000.0
    strategy_name: str = "vectorbt_backtest"
    strategy_version: str = "0.1.0"
    feature_set_version: str = "1.0.0"
    bar_seconds: int = 86400
    rebalance_every: int = 1
    top_n: int = 10
    output_root: Path = Path("data/backtest")


@dataclass(frozen=True)
class BacktestIntradayRequest:
    """``research backtest intraday``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    decision_time: tuple[str, ...]
    data_file: Path | None = None
    vendor: str = "file"
    initial_capital: float = 100_000.0
    strategy_name: str = "intraday_backtest"
    strategy_version: str = "0.1.0"
    feature_set_version: str = "1.0.0"
    model_version: str = "classical"
    universe_name: str = "intraday_research"
    dataset_id: tuple[str, ...] = ()
    output_root: Path = Path("data/backtest")


@dataclass(frozen=True)
class BacktestEvidenceAssertRequest:
    """``research backtest evidence assert``."""

    command: str
    manifest: Path


BacktestRequest = BacktestRunRequest | BacktestIntradayRequest | BacktestEvidenceAssertRequest


# --------------------------------------------------------------------------
# research-campaign
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignRunRequest:
    """``research-campaign run``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    model_version: str
    feature_set_version: str = "1.0.0"
    date_policy: str = "nyse-sessions"
    horizon_days: int = 21
    signal_type: str = "auto"
    output_root: Path | None = None
    train_xgboost: bool = False
    xgboost_search: str = "off"
    xgboost_device: str = "auto"
    xgboost_require_gpu: bool = False
    paper_source_weights_json: str = ""
    bar_seconds: int = 86400
    max_feature_age_days: int = 3
    train_window_days: int = 252
    test_window_days: int = 21
    step_days: int = 21
    purge_days: int = 21
    embargo_days: int = 0
    min_folds: int = 3
    min_oos_rolling_ic: float = 0.05
    min_ic_60d: float = 0.03
    # Fold-level streak — see AlphaEligibilityThresholds for the unit rationale.
    max_fold_negative_ic_streak: int = 2
    max_drawdown: float = -0.20
    min_slippage_adjusted_sharpe: float = 1.0
    slippage_bps_per_turnover: float = 10.0
    return_scale: float = 1.0
    campaign_portfolio_mode: str = "runtime-long-only"
    campaign_top_n: int = 10
    campaign_vol_target: float = 0.15
    campaign_vol_floor: float = 0.05
    campaign_vol_lookback_days: int = 63
    campaign_max_gross_exposure: float = 0.60
    campaign_min_cash_buffer: float = 0.05
    campaign_max_single_name_weight: float = 0.05
    campaign_max_daily_turnover: float = 0.20
    campaign_max_position_change: float = 0.05
    campaign_no_trade_band: float = 0.0
    campaign_rebalance_interval_days: int = 1
    max_calibration_age_days: float = 14.0
    require_calibration: bool = False
    feature_audit_mode: str = "shadow"
    feature_card_dir: Path | None = None
    feature_diagnostics: Path | None = None
    feature_family_file: Path | None = None
    source_data_manifest: Path | None = None
    text_prompt_version: str = ""
    attribution_horizons: tuple[int, ...] = (5, 10, 21)
    attribution_permutation_seed: int = 17
    attribution_permutation_count: int = 200
    attribution_correlation_threshold: float = 0.70
    min_null_qualified_features: int = 3
    feature_admission: Literal["passing", "all"] = "passing"
    min_admitted_features: int = 3
    model_feature: tuple[str, ...] | None = None
    fail_on_ineligible: bool = False


@dataclass(frozen=True)
class CampaignDiagnoseFeaturesRequest:
    """``research-campaign diagnose-features``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    feature_card_dir: Path
    feature_set_version: str = "paper-alpha-catalyst-v10"
    date_policy: str = "nyse-sessions"
    output_root: Path | None = None
    bar_seconds: int = 86400
    max_feature_age_days: int = 3
    slippage_bps_per_turnover: float = 10.0
    max_calibration_age_days: float = 14.0
    require_calibration: bool = False
    horizon_days: int = 21


@dataclass(frozen=True)
class CampaignAttributeFailuresRequest:
    """``research-campaign attribute-feature-failures``."""

    command: str
    contracts_file: str
    start: datetime
    end: datetime
    feature_card_dir: Path
    feature_set_version: str = "paper-alpha-catalyst-v10"
    date_policy: str = "nyse-sessions"
    output_root: Path | None = None
    bar_seconds: int = 86400
    max_feature_age_days: int = 3
    slippage_bps_per_turnover: float = 10.0
    max_calibration_age_days: float = 14.0
    require_calibration: bool = False
    horizons: tuple[int, ...] = (5, 10, 21)
    official_horizon_days: int = 21
    feature_family_file: Path = Path("infra/config/feature_families/paper-alpha-catalyst-v10.json")
    permutation_seed: int = 17
    permutation_count: int = 200
    correlation_threshold: float = 0.70


@dataclass(frozen=True)
class CampaignScreenRequest:
    """``research-campaign screen-{text,event,intraday}-candidates``."""

    command: str
    samples_file: Path
    sample_build_summary: Path
    sample_start: datetime | None = None
    sample_end: datetime | None = None
    output_root: Path | None = None
    screen_name: str = ""
    source_data_manifest: Path | None = None
    contracts_file: Path | None = None
    intraday_file: tuple[str, ...] = ()
    text_feature_set_version: str = "text-v5"
    promoted_feature_set_version: str = "paper-alpha-catalyst-v10"
    event_feature_set_version: str = "paper-alpha-event-reaction-v2"
    intraday_feature_set_version: str = "paper-alpha-intraday-microstructure-v2"
    candidate_family: str = ""
    candidate_set: str = "v10-alpha-quality"
    lookback_days: int = 21
    min_source_density: float = 0.05
    min_null_margin: float = 0.0
    min_ic_mean: float = 0.02
    min_icir: float = 0.10
    max_negative_ic_streak: int = 3
    min_passing_candidates: int = 3
    permutation_seed: int = 17
    permutation_count: int = 200


@dataclass(frozen=True)
class CampaignPromoteRequest:
    """``research-campaign promote-{text,event,intraday}-candidates``."""

    command: str
    main_screen: Path
    confirmation_screen: Path
    full_screen: Path
    feature_card_dir: Path
    feature_family_file: Path
    min_passing_candidates: int = 3
    output_root: Path | None = None
    screen_name: str = ""


CampaignRequest = (
    CampaignRunRequest
    | CampaignDiagnoseFeaturesRequest
    | CampaignAttributeFailuresRequest
    | CampaignScreenRequest
    | CampaignPromoteRequest
)


__all__ = [
    "AlphaRequest",
    "BacktestEvidenceAssertRequest",
    "BacktestIntradayRequest",
    "BacktestRequest",
    "BacktestRunRequest",
    "BoostingRequest",
    "CampaignAttributeFailuresRequest",
    "CampaignDiagnoseFeaturesRequest",
    "CampaignPromoteRequest",
    "CampaignRequest",
    "CampaignRunRequest",
    "CampaignScreenRequest",
    "FeatureAuditRequest",
    "FeaturesBackfillIntradayAlphaRequest",
    "FeaturesBackfillRequest",
    "FeaturesBuildSamplesRequest",
    "FeaturesRequest",
    "FeaturesRetentionRequest",
    "ModelRegistryRequest",
    "WalkForwardRequest",
]
