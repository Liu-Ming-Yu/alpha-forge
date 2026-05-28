"""Promotion-mode calculation for production-candidate reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import (
    PreflightCheck,
    ProductionProfile,
    PromotionMode,
    ReadinessState,
)

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def _compute_promotion_mode(
    *,
    profile: ProductionProfile,
    state: ReadinessState,
    checks: tuple[PreflightCheck, ...],
    clean_live_days: int,
    settings: PlatformSettings,
) -> tuple[PromotionMode, tuple[str, ...]]:
    blockers = tuple(
        check.name for check in checks if not check.passed and check.severity == "error"
    )
    if blockers or state != ReadinessState.READY:
        return PromotionMode.SHADOW_ONLY, blockers
    if profile == ProductionProfile.PAPER:
        return PromotionMode.PAPER_ENSEMBLE, ()
    if profile == ProductionProfile.LLM_LIVE_REHEARSAL:
        return PromotionMode.LLM_LIVE_REHEARSAL, ()
    # LIVE
    if clean_live_days >= 60 and float(settings.alpha.live_ramp_after_60d) > 0:
        return PromotionMode.LIVE_RAMP_AFTER_60D, ()
    if clean_live_days >= 20 and float(settings.alpha.live_ramp_after_20d) > 0:
        return PromotionMode.LIVE_RAMP_AFTER_20D, ()
    return PromotionMode.LIVE_RAMP_INITIAL, ()
