"""Production-candidate promotion domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from quant_platform.core.domain.production.readiness import (
    PreflightCheck,
    ProductionProfile,
    ProductionReadinessReport,
    ReadinessState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.production.performance import SignalGateStatus
    from quant_platform.core.domain.production.prediction import ForecastEvidence


class PromotionMode(StrEnum):
    """Permitted operating mode after a production-candidate evaluation.

    The mode ladder is monotonic: each higher rung requires every lower
    rung's evidence plus its own additional gates.  ``shadow_only`` is the
    safe fallback whenever any error-severity check fails.
    """

    SHADOW_ONLY = "shadow_only"
    PAPER_ENSEMBLE = "paper_ensemble"
    LLM_LIVE_REHEARSAL = "llm_live_rehearsal"
    LIVE_RAMP_INITIAL = "live_ramp_initial"
    LIVE_RAMP_AFTER_20D = "live_ramp_after_20d"
    LIVE_RAMP_AFTER_60D = "live_ramp_after_60d"


@dataclass(frozen=True)
class ProductionCandidateReport:
    """Aggregated production-candidate evaluation.

    Wraps a ``ProductionReadinessReport`` and adds research-campaign,
    multi-source signal-gate, and V2 orchestrator evidence to compute
    ``next_allowed_mode`` — the highest operating rung the runtime is
    currently cleared for.

    ``promotion_blockers`` lists the subset of failing checks that prevent
    advancing past ``next_allowed_mode``.
    """

    profile: ProductionProfile
    generated_at: datetime
    state: ReadinessState
    next_allowed_mode: PromotionMode
    promotion_blockers: tuple[str, ...]
    checks: tuple[PreflightCheck, ...]
    readiness: ProductionReadinessReport
    campaign_manifest_path: str | None = None
    campaign_manifest: Mapping[str, object] | None = None
    representative_signal_gate_status: SignalGateStatus | None = None
    signal_gate_statuses: tuple[SignalGateStatus, ...] = ()
    forecast_evidence: tuple[ForecastEvidence, ...] = ()

    @property
    def passed(self) -> bool:
        return self.state == ReadinessState.READY and not self.promotion_blockers

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(
            check for check in self.checks if not check.passed and check.severity == "error"
        )
