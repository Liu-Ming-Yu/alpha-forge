"""LLM-live rehearsal checks for production-candidate gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import PreflightCheck
from quant_platform.services.governance_service.llm_live_startup import (
    LLM_LIVE_MAX_INITIAL_CAP,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings


async def llm_shadow_paper_parity_check(
    evidence_repo: object,
    *,
    as_of: datetime,
) -> PreflightCheck:
    status_fn = getattr(evidence_repo, "shadow_paper_parity_status", None)
    if status_fn is None:
        return PreflightCheck(
            name="llm_live_shadow_paper_parity_passed",
            passed=False,
            detail="performance repository does not expose shadow-paper parity evidence",
        )
    status = await status_fn(
        "text",
        "text",
        as_of=as_of,
        min_trading_days=20,
        max_target_weight_diff_bps=1.0,
    )
    return PreflightCheck(
        name="llm_live_shadow_paper_parity_passed",
        passed=status.passed,
        detail=(
            f"trading_days={status.trading_days} observations={status.observations} "
            f"max_target_weight_diff_bps={status.max_target_weight_diff_bps:.4f} "
            f"missing_instruments={status.missing_instruments} "
            f"order_side_mismatches={status.order_side_mismatches} "
            f"blockers={'; '.join(status.blockers)}"
        ),
    )


def llm_live_rehearsal_config_checks(
    settings: PlatformSettings,
    sources: tuple[str, ...],
) -> list[PreflightCheck]:
    return [
        PreflightCheck(
            name="llm_live_rehearsal_enabled",
            passed=settings.llm.live_mode_enabled and settings.llm.live_rehearsal_enabled,
            detail=(
                f"live_mode_enabled={settings.llm.live_mode_enabled} "
                f"live_rehearsal_enabled={settings.llm.live_rehearsal_enabled}"
            ),
        ),
        PreflightCheck(
            name="llm_live_rehearsal_ensemble_mode_live",
            passed=settings.alpha.ensemble_mode == "live",
            detail=f"ensemble_mode={settings.alpha.ensemble_mode}",
        ),
        PreflightCheck(
            name="llm_live_rehearsal_paper_broker",
            passed=settings.broker.paper_trading,
            detail=f"paper_trading={settings.broker.paper_trading}",
        ),
        PreflightCheck(
            name="llm_live_rehearsal_replay_only",
            passed=settings.llm.replay_only_live,
            detail=f"replay_only_live={settings.llm.replay_only_live}",
        ),
        PreflightCheck(
            name="llm_live_rehearsal_fail_closed_enabled",
            passed=settings.alpha.fail_closed_on_promoted_source_error,
            detail=(
                "promoted source failures block the cycle"
                if settings.alpha.fail_closed_on_promoted_source_error
                else "promoted source failures may be ignored"
            ),
        ),
        PreflightCheck(
            name="llm_live_rehearsal_text_source_positive_weight",
            passed=(settings.alpha.source_weights.get("text", 0.0) > 0 and "text" in sources),
            detail=(
                f"text_weight={settings.alpha.source_weights.get('text', 0.0)} "
                f"sources={list(sources)}"
            ),
        ),
        _llm_live_rehearsal_source_weights_check(settings),
        PreflightCheck(
            name="llm_live_rehearsal_initial_cap_conservative",
            passed=(
                settings.alpha.max_non_classical_weight <= float(LLM_LIVE_MAX_INITIAL_CAP)
                and float(settings.alpha.live_ramp_initial) <= float(LLM_LIVE_MAX_INITIAL_CAP)
            ),
            detail=(
                f"max_non_classical_weight={settings.alpha.max_non_classical_weight} "
                f"live_ramp_initial={settings.alpha.live_ramp_initial} "
                f"max_allowed={LLM_LIVE_MAX_INITIAL_CAP}"
            ),
        ),
    ]


def _llm_live_rehearsal_source_weights_check(settings: PlatformSettings) -> PreflightCheck:
    weights = settings.alpha.source_weights
    expected = {
        "classical": 0.99,
        "text": 0.01,
        "xgboost": 0.0,
        "event": 0.0,
        "intraday": 0.0,
    }
    explicit = set(expected).issubset(weights)
    pinned = explicit and all(
        abs(float(weights.get(name, -1.0)) - value) <= 1e-12 for name, value in expected.items()
    )
    return PreflightCheck(
        name="llm_live_rehearsal_source_weights_pinned",
        passed=pinned,
        detail=f"expected={expected} configured={weights} explicit_all_sources={explicit}",
    )


__all__ = [
    "llm_live_rehearsal_config_checks",
    "llm_shadow_paper_parity_check",
]
