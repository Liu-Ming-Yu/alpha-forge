"""Runtime limit checks for live-LLM startup governance."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import PreflightCheck
from quant_platform.services.governance_service.llm_live_startup.constants import (
    LLM_LIVE_MAX_INITIAL_CAP,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def build_runtime_limit_checks(settings: PlatformSettings) -> list[PreflightCheck]:
    """Return live-LLM cap and provider-budget checks."""
    return [
        _live_cap_check(settings),
        _provider_budget_check(settings),
    ]


def _live_cap_check(settings: PlatformSettings) -> PreflightCheck:
    cap = Decimal(str(settings.alpha.max_non_classical_weight))
    initial = Decimal(str(settings.alpha.live_ramp_initial))
    passed = cap <= LLM_LIVE_MAX_INITIAL_CAP and initial <= LLM_LIVE_MAX_INITIAL_CAP
    return PreflightCheck(
        name="llm_live_initial_cap_conservative",
        passed=passed,
        detail=(
            f"max_non_classical_weight={cap} live_ramp_initial={initial} "
            f"max_allowed={LLM_LIVE_MAX_INITIAL_CAP}"
        ),
    )


def _provider_budget_check(settings: PlatformSettings) -> PreflightCheck:
    passed = (
        settings.llm.max_request_latency_seconds <= settings.llm.timeout_seconds
        and settings.llm.max_daily_calls >= 1
        and settings.llm.estimated_cost_per_call_usd <= settings.llm.max_daily_estimated_cost_usd
        and settings.llm.replay_only_live
    )
    return PreflightCheck(
        name="llm_provider_runtime_limits_configured",
        passed=passed,
        detail=(
            f"max_request_latency_seconds={settings.llm.max_request_latency_seconds} "
            f"timeout_seconds={settings.llm.timeout_seconds} "
            f"max_daily_calls={settings.llm.max_daily_calls} "
            f"max_daily_estimated_cost_usd={settings.llm.max_daily_estimated_cost_usd} "
            f"estimated_cost_per_call_usd={settings.llm.estimated_cost_per_call_usd} "
            f"replay_only_live={settings.llm.replay_only_live}"
        ),
    )
