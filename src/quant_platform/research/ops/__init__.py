"""Research, feature, and model-registry operation composition."""

from __future__ import annotations

from quant_platform.research.backtesting.ops import (
    _backtest_evidence,
    _backtest_intraday_impl,
    _backtest_run,
    _backtest_run_impl,
)
from quant_platform.research.campaign.feature_audit_ops import (
    _run_campaign_feature_audits,
)
from quant_platform.research.campaign.inputs import (
    _observed_slippage_bps,
    _parse_paper_source_weights,
)
from quant_platform.research.campaign.ops import (
    _research_campaign,
    _research_campaign_run,
)
from quant_platform.research.common import (
    _BACKTEST_WARMUP_DAYS,
    _build_samples_to_path,
    _instrument_lookup_from_contracts,
    _json_default,
    _latest_calibration_artifact,
    _load_calibration_recommendation_bps,
    _load_instrument_contracts,
    _parse_intraday_decision_times,
    _require_durable_research_inputs,
    _samples_result_payload,
    _verify_postgres_schema_if_configured,
)
from quant_platform.research.features.ops import (
    _features,
    _features_audit,
    _features_audit_assert,
    _features_audit_retire,
    _features_audit_run,
    _features_audit_status,
    _features_backfill,
    _features_build_samples,
    _features_retention,
)
from quant_platform.research.misc_ops import (
    _factors_calibrate,
    _tearsheet,
)
from quant_platform.research.modeling.model_ops import _boosting, _model_registry
from quant_platform.research.modeling.walk_forward_ops import _walk_forward

__all__ = [
    "_BACKTEST_WARMUP_DAYS",
    "_backtest_evidence",
    "_backtest_intraday_impl",
    "_backtest_run",
    "_backtest_run_impl",
    "_boosting",
    "_build_samples_to_path",
    "_factors_calibrate",
    "_features",
    "_features_audit",
    "_features_audit_assert",
    "_features_audit_retire",
    "_features_audit_run",
    "_features_audit_status",
    "_features_backfill",
    "_features_build_samples",
    "_features_retention",
    "_instrument_lookup_from_contracts",
    "_json_default",
    "_latest_calibration_artifact",
    "_load_calibration_recommendation_bps",
    "_load_instrument_contracts",
    "_model_registry",
    "_observed_slippage_bps",
    "_parse_intraday_decision_times",
    "_parse_paper_source_weights",
    "_require_durable_research_inputs",
    "_research_campaign",
    "_research_campaign_run",
    "_run_campaign_feature_audits",
    "_samples_result_payload",
    "_tearsheet",
    "_verify_postgres_schema_if_configured",
    "_walk_forward",
]
