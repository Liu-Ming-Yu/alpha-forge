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
    #: Drawdown-conditioned streak relaxation (ADR-004 Option D). When set, a
    #: streak above ``max_fold_negative_ic_streak`` (the strict floor) is
    #: tolerated up to this cap **only if** the drawdown during the worst streak
    #: stayed inside ``streak_containment_max_drawdown``; otherwise the strict
    #: floor applies. When ``None`` (default) the streak gate is the plain
    #: ``streak <= floor`` check and the ``max_drawdown_during_worst_streak``
    #: metric is never read — so callers predating the field are unaffected.
    #: Requires the metrics dict to carry ``max_drawdown_during_worst_streak``
    #: whenever a streak exceeds the floor and this is non-``None``.
    max_fold_negative_ic_streak_if_dd_contained: int | None = None
    #: Within-worst-streak drawdown bound for the relaxation above. It must be
    #: **tighter** than ``max_drawdown``: the within-streak drawdown is a subset
    #: of the full-run drawdown, so reusing the full-run bound would make the
    #: condition redundant (it could never independently fail). A tight bound
    #: instead distinguishes "the construction absorbed the IC inversion" (small
    #: within-streak DD — relaxation earned) from "the episode caused the loss"
    #: (within-streak DD near the full bound — relaxation forfeit, even if the
    #: full-run DD gate still passes). ``None`` falls back to ``max_drawdown``.
    streak_containment_max_drawdown: float | None = None
    max_drawdown: float = -0.20
    min_slippage_adjusted_sharpe: float = 1.0
    #: Minimum bootstrap 5th-percentile IC — the robustness gate that *replaces*
    #: the brittle negative-IC-streak count for portfolio candidates (ADR-004
    #: 2026-05-29). The held-out calibration proved the streak metric is not
    #: OOS-stable (calibration-window streak 3 vs validation-window streak 7 on
    #: the same arm), so a fixed streak cap cannot generalise. ``bootstrap_ic_p05``
    #: — the 5th percentile of the block-bootstrapped fold-IC distribution —
    #: tests the same concern ("is the predictive power *reliably* positive?")
    #: with a statistically-grounded, regime-inclusive metric (the bootstrap
    #: resamples span the crash episodes). ``> 0`` means "95% confident the IC is
    #: positive". ``None`` (default) leaves the gate off, so callers/categories
    #: predating the field are unaffected.
    min_bootstrap_ic_p05: float | None = None
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
#: negative-IC stretches without translating them into catastrophic P&L.
#:
#: Streak gate redesign (ADR-004, 2026-05-29). History: ``streak <= 4`` (v1) →
#: drawdown-conditioned ``floor 2 / cap 6`` (v2). After the dollar-volume scoring
#: fix (ADR-011) the held-out calibration on corrected evidence was decisive:
#: the negative-IC-streak metric is **not OOS-stable** — the *same arm* shows a
#: calibration-window streak of 3 and a validation-window streak of 7 (the
#: 2024-summer crash episode falls entirely out-of-sample), and the only
#: cal==val-stable cap is 9 (≈ no gate). A run-length count therefore cannot be
#: a principled discriminator on this universe/label.
#:
#: v3 replaces it with a **bootstrap-IC significance gate**: the streak count is
#: demoted to a loose catastrophic backstop at the one OOS-stable value (``9``;
#: it never binds for a sane arm), and the real robustness gate becomes
#: ``min_bootstrap_ic_p05 > 0`` — the predictive power must be *statistically*
#: positive (5th percentile of the block-bootstrapped fold-IC distribution above
#: zero). That tests the same thing the streak gate intended ("is the alpha
#: reliably positive across regimes?") but with a metric that is regime-inclusive
#: by construction and does not depend on where an episode lands in the window.
#: It is strict, not a rubber stamp: on the corrected A–N evidence it admits
#: D (p05 +0.018) and N (+0.011) but rejects the GBDT-rank arm J (p05 −0.006)
#: whose Sharpe-1.28 is driven by a single crash episode, not robust ranking.
#:
#: This is the eligibility-threshold separation called out in ADR-003; it lets
#: governance distinguish "the alpha is dead" (baselines must clear) from "the
#: alpha is reliably positive" (candidates can clear).
PORTFOLIO_CANDIDATE_THRESHOLDS: AlphaEligibilityThresholds = AlphaEligibilityThresholds(
    name="portfolio_candidate_v3",
    # Loose catastrophic backstop at the only OOS-stable cap (held-out
    # calibration); the binding robustness gate is min_bootstrap_ic_p05 below.
    max_fold_negative_ic_streak=9,
    max_drawdown=-0.10,
    # The redesigned robustness gate: IC must be statistically-positive (95%).
    min_bootstrap_ic_p05=0.0,
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
