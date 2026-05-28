"""Forecast-evidence checks for live-LLM startup governance."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import ForecastEvidence, PreflightCheck
from quant_platform.services.governance_service.llm_live_startup.paths import (
    expected_text_feature_schema_hash,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def prediction_schema_hash_check(
    settings: PlatformSettings,
    evidence: ForecastEvidence,
) -> PreflightCheck:
    """Return the text prediction schema-hash check."""
    expected = expected_text_feature_schema_hash(settings)
    schema_hashes = tuple(evidence.feature_schema_hashes)
    return PreflightCheck(
        name="llm_live_prediction_schema_hash_matches",
        passed=schema_hashes == (expected,),
        detail=json.dumps({"expected": expected, "observed": schema_hashes}, sort_keys=True),
    )


def prediction_source_horizon_check(evidence: ForecastEvidence) -> PreflightCheck:
    """Return the text prediction source/horizon check."""
    return PreflightCheck(
        name="llm_live_prediction_source_horizon_21d",
        passed=evidence.source == "text" and evidence.horizon == "21d",
        detail=f"source={evidence.source} horizon={evidence.horizon}",
    )
