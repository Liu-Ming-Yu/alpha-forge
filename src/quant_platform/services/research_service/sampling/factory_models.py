"""Typed DTOs for research campaign evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from quant_platform.services.research_service.sampling.arm_category import ArmCategory


@dataclass(frozen=True)
class AlphaEligibilityThresholds:
    """Promotion thresholds for aggressive paper/shadow research candidates.

    Unit note on ``max_fold_negative_ic_streak``
    --------------------------------------------
    The streak is measured on fold-level mean ICs, not on daily ICs. Daily ICs
    on a multi-day forward-return horizon are overlapping labels — one bad
    window produces a long correlated negative run even when the underlying
    signal is healthy. Fold ICs are mostly independent (purge + embargo) so a
    short streak is a real "consecutive periods of failure" signal.

    Default ``2`` allows up to two consecutive negative folds, which a
    marginally-positive signal (E[IC]≈0.01–0.02 per fold) passes comfortably
    while a zero-mean noise series fails the gate with high probability.

    Backwards-incompat (2026-05-25)
    -------------------------------
    Field was renamed from ``max_negative_ic_streak`` (default 3) to
    ``max_fold_negative_ic_streak`` (default 2). The old name and old default
    were calibrated against daily-IC streaks, and silently became a much
    stricter gate after the streak source changed to fold IC. Construct with
    the new field name; saved evidence/dashboards comparing the old key vs the
    new one are not comparable.

    Per-category instances
    ----------------------
    Default-constructed thresholds (``AlphaEligibilityThresholds()``) are
    calibrated for **research_ranker_baseline** arms — signed-rank weights
    with no risk controls, naturally producing wide drawdowns. Production-
    candidate arms (long-only with name/gross caps + monthly rebal + dial)
    have a different risk profile and should use
    :data:`PORTFOLIO_CANDIDATE_THRESHOLDS` instead; see ADR-003 for the
    framing. The ``name`` field travels with each instance so audit-trail
    readers can identify which set was applied without diffing the values.
    """

    min_oos_rolling_ic: float = 0.05
    min_ic_60d: float = 0.03
    max_fold_negative_ic_streak: int = 2
    max_drawdown: float = -0.20
    min_slippage_adjusted_sharpe: float = 1.0
    #: Human-readable label for this threshold set. Serialised into the
    #: evidence JSON so a future auditor can identify "which named set
    #: was applied" without re-deriving it from the numeric values.
    #:
    #: The default is set to ``"research_ranker_baseline_v1"`` so that
    #: ``AlphaEligibilityThresholds()`` and the named
    #: :data:`RESEARCH_RANKER_BASELINE_THRESHOLDS` instance produce
    #: identical evidence strings — both construction paths yield the
    #: same governance contract on the audit trail. Pre-PR-71 evidence
    #: that was produced before this field existed lacks the ``name``
    #: key entirely; pre-this-commit evidence may carry the legacy
    #: ``"default_strict"`` string.
    name: str = "research_ranker_baseline_v1"


#: Default thresholds for **research_ranker_baseline** arms. Same numeric
#: values as ``AlphaEligibilityThresholds()`` (the legacy default); the
#: named instance just gives audit trails an explicit label.
#:
#: Baselines are signed-rank weights with no risk controls — they
#: naturally produce 15-17% drawdowns and can run negative-IC streaks
#: of 4-7 folds without it being catastrophic. The gate is strict
#: precisely because baselines should never PASS eligibility unless
#: they would *also* pass with risk controls. They are diagnostic
#: tools measuring whether features rank returns, not portfolios.
RESEARCH_RANKER_BASELINE_THRESHOLDS: AlphaEligibilityThresholds = AlphaEligibilityThresholds(
    name="research_ranker_baseline_v1",
)


#: Default thresholds for **portfolio_candidate** arms (long-only top-N
#: with per-name + gross caps + monthly rebal). The construction absorbs
#: negative-IC stretches without translating them into catastrophic
#: P&L, so the streak gate widens from 2 to 4 folds (~84 trading days).
#: In exchange, the drawdown gate tightens from -20% to -10%: if a
#: tagged-candidate's construction misbehaves and DD blows past -10%,
#: the looser streak gate doesn't help — the DD gate fails first.
#: Together these encode "we trust the construction iff it actually
#: protects you."
#:
#: This is the eligibility-threshold separation called out in
#: ADR-003 ("per-category eligibility thresholds"); it lets governance
#: distinguish "the alpha is dead" (baselines must clear) from
#: "the alpha is noisy but the construction handles it" (candidates
#: can clear).
PORTFOLIO_CANDIDATE_THRESHOLDS: AlphaEligibilityThresholds = AlphaEligibilityThresholds(
    name="portfolio_candidate_v1",
    max_fold_negative_ic_streak=4,
    max_drawdown=-0.10,
)


#: Lookup table: :data:`ArmCategory` → default threshold set. Scripts
#: that produce evidence for multiple arm categories (e.g. the latest-
#: stack backtest) dispatch through this mapping so the right gate
#: applies to each arm. The Literal-typed key means a typo at the
#: call site fails at type-check time, not at dispatch.
#:
#: Wrapped in :class:`MappingProxyType` so this is a read-only view —
#: a future caller can't splice in a custom threshold set at runtime
#: and defeat the lookup-is-canonical contract. To extend the
#: vocabulary, add a value to :data:`ArmCategory`, define a new
#: :class:`AlphaEligibilityThresholds` instance for it, and register
#: it here. All three edits land in this file.
THRESHOLDS_BY_ARM_CATEGORY: Mapping[ArmCategory, AlphaEligibilityThresholds] = MappingProxyType(
    {
        "research_ranker_baseline": RESEARCH_RANKER_BASELINE_THRESHOLDS,
        "portfolio_candidate": PORTFOLIO_CANDIDATE_THRESHOLDS,
    }
)


@dataclass(frozen=True)
class WalkForwardEvidence:
    """Research evidence bundle for one candidate model.

    The bundle now carries enough evidence for a governed promotion gate:

    - ``daily_returns`` / ``daily_ics`` summarise the OOS series.
    - ``daily_turnover`` is the per-day signed-rank turnover used to apply
      slippage costs and report turnover-adjusted Sharpe.
    - ``feature_stability`` is the mean absolute change in selected weights
      from one fold to the next, scored per feature.
    - ``bootstrap_ic_ci`` is a 95 percent bootstrap confidence interval
      for the rolling cross-sectional IC.
    - ``attribution`` groups average IC and average return by metadata key
      (e.g. sector, regime) so reviewers can spot regime-specific failures.
    - ``slippage_bps_per_turnover`` records the slippage assumption that
      penalised the OOS returns; the production-candidate gate compares it
      against the calibrated value pulled from stored paper fills.
    """

    run_id: uuid.UUID
    model_version: str
    feature_set_version: str
    folds: tuple[dict[str, object], ...]
    selected_weights: Mapping[str, float]
    daily_returns: tuple[float, ...]
    daily_ics: tuple[tuple[str, float], ...]
    metrics: Mapping[str, float]
    eligibility: Mapping[str, object]
    artifact_root: Path | None = None
    daily_turnover: tuple[float, ...] = ()
    feature_stability: Mapping[str, float] = field(default_factory=dict)
    bootstrap_ic_ci: tuple[float, float] = (0.0, 0.0)
    attribution: Mapping[str, Mapping[str, dict[str, float]]] = field(default_factory=dict)
    slippage_bps_per_turnover: float = 0.0
    portfolio_config: Mapping[str, object] = field(default_factory=dict)
    portfolio_diagnostics: Mapping[str, object] = field(default_factory=dict)
    drawdown_diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchCampaignManifest:
    """Machine-readable paper research campaign index."""

    run_id: uuid.UUID
    created_at: datetime
    model_version: str
    feature_set_version: str
    passed: bool
    metrics: Mapping[str, float]
    eligibility: Mapping[str, object]
    artifacts: Mapping[str, str | None]
    selected_weights: Mapping[str, float]
    paper_source_weights: Mapping[str, float]
    git_commit: str
    next_allowed_paper_mode: str
