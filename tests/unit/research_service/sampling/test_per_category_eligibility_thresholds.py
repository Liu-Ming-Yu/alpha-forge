"""Pins the per-category eligibility threshold sets.

ADR-003 named the gap: ``AlphaEligibilityThresholds`` was a single
global dataclass applied uniformly to research_ranker_baseline arms
(signed-rank with no risk controls, naturally wide drawdowns) and
portfolio_candidate arms (long-only with name/gross caps and a
streak dial that keep drawdowns inside −5%). One threshold cannot
calibrate both — that's bad governance.

These tests pin the two named threshold sets plus the lookup table
the scripts use to dispatch by ``ArmCategory``. The actual gate
behaviour is exercised end-to-end by the latest-stack rerun; the
unit tests here cover:

1. The named instances exist and have sensible value relationships
   to each other (portfolio_candidate gives looser streak in
   exchange for tighter drawdown).
2. The lookup covers every ``ArmCategory`` value used in the
   latest-stack registry — a typo on either side surfaces here.
3. A simulated G-style metrics dict (Sharpe 1.09, DD −4.2%, streak
   4) passes the portfolio_candidate gate but fails the research_-
   ranker_baseline gate. This is the smallest test that pins the
   new governance contract: "the same model can be ineligible as a
   baseline but eligible as a candidate, because the construction
   is doing protective work."
"""

from __future__ import annotations

import sys
from pathlib import Path

# The latest-stack script lives in ``scripts/`` (not on the default
# import path). Match the existing pattern in
# ``test_backtest_latest_stack_sample_builder.py``: add the project
# root to sys.path at module load so ``from scripts.backtest_latest_stack
# import ARM_SPECS`` resolves cleanly inside test methods that need it.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quant_platform.services.research_service.sampling.eligibility import (  # noqa: E402
    eligibility,
)
from quant_platform.services.research_service.sampling.factory_models import (  # noqa: E402
    PORTFOLIO_CANDIDATE_THRESHOLDS,
    RESEARCH_RANKER_BASELINE_THRESHOLDS,
    THRESHOLDS_BY_ARM_CATEGORY,
    AlphaEligibilityThresholds,
)

# -- 1. Named instances exist and value relationships make sense -----------


class TestThresholdValueRelationships:
    """The portfolio_candidate set trades streak laxity for drawdown
    strictness. If a future tune-up breaks that trade — e.g. loosens
    both — it would invert the governance contract."""

    def test_baseline_uses_strict_streak_gate(self) -> None:
        # 2 is the audit-calibrated tight gate; baselines should
        # stay at it.
        assert RESEARCH_RANKER_BASELINE_THRESHOLDS.max_fold_negative_ic_streak == 2

    def test_baseline_uses_legacy_drawdown_gate(self) -> None:
        # −20% is the original gate; baselines naturally produce
        # 15-17% drawdowns and need this room.
        assert RESEARCH_RANKER_BASELINE_THRESHOLDS.max_drawdown == -0.20

    def test_portfolio_candidate_streak_is_looser_than_baseline(self) -> None:
        # v3 (ADR-004 2026-05-29): the negative-IC-streak count proved not
        # OOS-stable (held-out calibration: same arm shows cal streak 3 / val
        # streak 7), so the candidate gate demotes it to a loose catastrophic
        # backstop at the one OOS-stable cap (9) and gates robustness on
        # bootstrap-IC significance instead. The backstop is still looser than
        # the baseline's strict floor.
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.max_fold_negative_ic_streak
            > RESEARCH_RANKER_BASELINE_THRESHOLDS.max_fold_negative_ic_streak
        )
        # The v2 drawdown-conditioned relaxation is superseded by the bootstrap
        # gate: the candidate carries the robustness gate, the baseline does not.
        assert PORTFOLIO_CANDIDATE_THRESHOLDS.min_bootstrap_ic_p05 == 0.0
        assert RESEARCH_RANKER_BASELINE_THRESHOLDS.min_bootstrap_ic_p05 is None

    def test_portfolio_candidate_drawdown_is_tighter_than_baseline(self) -> None:
        # Tighter drawdown: if a tagged-candidate's construction
        # misbehaves and DD exceeds −10%, we want to catch it —
        # the construction was supposed to cap exposure.
        # ``max_drawdown`` is negative; ``tighter`` means a smaller
        # |magnitude| / a less-negative number.
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.max_drawdown
            > RESEARCH_RANKER_BASELINE_THRESHOLDS.max_drawdown
        )

    def test_ic_gates_are_identical_across_categories(self) -> None:
        # IC strength is a model-quality property; the construction
        # doesn't change what it means for the alpha to predict.
        # Both categories must clear the same IC bar.
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.min_oos_rolling_ic
            == RESEARCH_RANKER_BASELINE_THRESHOLDS.min_oos_rolling_ic
        )
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.min_ic_60d
            == RESEARCH_RANKER_BASELINE_THRESHOLDS.min_ic_60d
        )

    def test_sharpe_gate_identical_across_categories(self) -> None:
        # Sharpe gate is post-cost; both categories must beat the
        # same break-even.
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.min_slippage_adjusted_sharpe
            == RESEARCH_RANKER_BASELINE_THRESHOLDS.min_slippage_adjusted_sharpe
        )

    def test_baseline_and_candidate_have_distinct_names(self) -> None:
        # The ``name`` field rides in the evidence JSON so audit
        # trails can identify which set was applied. Distinct names
        # for distinct gate calibrations are the entire point.
        assert RESEARCH_RANKER_BASELINE_THRESHOLDS.name != PORTFOLIO_CANDIDATE_THRESHOLDS.name

    def test_default_constructor_matches_baseline_set_exactly(self) -> None:
        # Backward compat: existing callers using
        # ``AlphaEligibilityThresholds()`` get values AND the name of
        # the named baseline set, so the two construction paths
        # produce identical evidence strings. Pre-this-PR evidence
        # may carry a legacy ``"default_strict"`` name; new evidence
        # has the baseline-v1 name unified across both paths.
        default = AlphaEligibilityThresholds()
        assert default == RESEARCH_RANKER_BASELINE_THRESHOLDS, (
            "AlphaEligibilityThresholds() and RESEARCH_RANKER_BASELINE_THRESHOLDS "
            "must be value-equal so audit trails can't divide them. If the "
            "default ever diverges, downstream evidence will carry two "
            "different names for what is structurally the same gate."
        )


# -- 2. Lookup table covers every category ---------------------------------


class TestThresholdsByArmCategory:
    """The lookup is keyed by string to keep ``factory_models`` free
    of script-specific imports (``ArmCategory`` lives in the latest-
    stack script). Tests here pin that the strings match what the
    script actually emits."""

    def test_lookup_contains_research_ranker_baseline(self) -> None:
        assert "research_ranker_baseline" in THRESHOLDS_BY_ARM_CATEGORY
        assert (
            THRESHOLDS_BY_ARM_CATEGORY["research_ranker_baseline"]
            is RESEARCH_RANKER_BASELINE_THRESHOLDS
        )

    def test_lookup_contains_portfolio_candidate(self) -> None:
        assert "portfolio_candidate" in THRESHOLDS_BY_ARM_CATEGORY
        assert THRESHOLDS_BY_ARM_CATEGORY["portfolio_candidate"] is PORTFOLIO_CANDIDATE_THRESHOLDS

    def test_lookup_keys_match_latest_stack_arm_categories(self) -> None:
        # The ``ArmCategory`` Literal defines the canonical category
        # vocabulary (now shared via
        # ``services.research_service.sampling.arm_category``). Every
        # value that appears on an ``ArmSpec`` in the latest-stack
        # registry MUST have a threshold set, otherwise the worker
        # dispatch raises ``KeyError`` at runtime — the loud-fail
        # design — but we'd rather pin completeness here.
        from scripts.backtest_latest_stack import ARM_SPECS  # noqa: PLC0415

        for spec in ARM_SPECS:
            assert spec.category in THRESHOLDS_BY_ARM_CATEGORY, (
                f"Arm {spec.cli_alias} has category {spec.category!r} but "
                "no threshold set is registered for it. Add one to "
                "THRESHOLDS_BY_ARM_CATEGORY or rename the category."
            )


# -- 3. The G-shaped governance contract -----------------------------------


class TestPerCategoryGateSeparation:
    """Pins the v3 governance contract (ADR-004 2026-05-29).

    The eligible lead on the corrected (normfixed) A–N evidence is D-shaped
    (Sharpe > 1, statistically-positive IC). It must:
    * FAIL the research_ranker_baseline gate (its streak exceeds the strict
      floor 2), and
    * PASS the portfolio_candidate gate (streak within the loose backstop 9,
      bootstrap-IC significantly positive, Sharpe and DD inside bounds).
    And a J-shaped arm (highest Sharpe of all, but its edge is one crash
    episode so the IC is not robustly positive) must FAIL the candidate gate on
    the bootstrap-IC robustness check — the whole point of the v3 redesign.
    """

    def _candidate_lead_metrics(self) -> dict[str, float]:
        # D-shaped: the eligible lead on the corrected A–N evidence.
        return {
            "oos_rolling_ic": 0.162,
            "ic_60d": 0.057,
            "fold_negative_ic_streak": 7.0,
            "max_drawdown": -0.033,
            "bootstrap_ic_p05": 0.018,
            "slippage_adjusted_sharpe": 1.091,
        }

    def _episode_trader_metrics(self) -> dict[str, float]:
        # J-shaped: highest Sharpe of any arm, but its edge is one crash episode,
        # so the bootstrapped fold-IC's 5th percentile is negative — the IC is
        # not statistically positive across regimes.
        return {
            "oos_rolling_ic": 0.268,
            "ic_60d": 0.053,
            "fold_negative_ic_streak": 9.0,
            "max_drawdown": -0.035,
            "bootstrap_ic_p05": -0.006,
            "slippage_adjusted_sharpe": 1.280,
        }

    def test_lead_fails_research_ranker_baseline_gate(self) -> None:
        result = eligibility(self._candidate_lead_metrics(), RESEARCH_RANKER_BASELINE_THRESHOLDS)
        assert result["passed"] is False
        # Baseline carries no bootstrap gate; it rejects the lead on the strict
        # streak floor (7 > 2).
        failing = [c for c in result["checks"] if not c["passed"]]
        assert any(c["name"] == "fold_negative_ic_streak" for c in failing)

    def test_lead_passes_portfolio_candidate_gate(self) -> None:
        result = eligibility(self._candidate_lead_metrics(), PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert result["passed"] is True, (
            "The D-shaped lead MUST pass the v3 portfolio_candidate gate; if "
            "this fails, a threshold value or the gate logic regressed."
        )
        for check in result["checks"]:
            assert check["passed"], f"check {check['name']!r} unexpectedly failed: {check}"

    def test_episode_trader_rejected_by_bootstrap_gate(self) -> None:
        # The v3 contract: a high-Sharpe arm whose IC is not statistically
        # positive is rejected. bootstrap_ic_p05 is the binding robustness gate
        # that replaced the OOS-unstable streak count.
        result = eligibility(self._episode_trader_metrics(), PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert result["passed"] is False
        failing = [c for c in result["checks"] if not c["passed"]]
        assert len(failing) == 1
        assert failing[0]["name"] == "bootstrap_ic_p05"

    def test_candidate_with_high_drawdown_still_blocked_by_dd_gate(self) -> None:
        # Defensive: a portfolio_candidate that produces a -15% DD (worse than
        # the -10% gate) is rejected even with a within-backstop streak and a
        # significantly-positive IC. The construction-trust contract is two-way.
        broken_candidate = {
            "oos_rolling_ic": 0.10,
            "ic_60d": 0.05,
            "fold_negative_ic_streak": 1.0,
            "max_drawdown": -0.15,  # WORSE than candidate's −0.10 gate
            "bootstrap_ic_p05": 0.02,
            "slippage_adjusted_sharpe": 1.5,
        }
        result = eligibility(broken_candidate, PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert result["passed"] is False
        failing = [c for c in result["checks"] if not c["passed"]]
        assert len(failing) == 1
        assert failing[0]["name"] == "max_drawdown"

    def test_loose_streak_does_not_help_baseline(self) -> None:
        # A research_ranker_baseline with streak=4 is rejected under its own
        # (strict floor 2) gate even though that streak is within the candidate's
        # loose backstop. Categories don't share thresholds — that's the point.
        metrics_with_streak_4 = {
            "oos_rolling_ic": 0.10,
            "ic_60d": 0.05,
            "fold_negative_ic_streak": 4.0,
            "max_drawdown": -0.05,
            "bootstrap_ic_p05": 0.02,
            "slippage_adjusted_sharpe": 1.5,
        }
        baseline_result = eligibility(metrics_with_streak_4, RESEARCH_RANKER_BASELINE_THRESHOLDS)
        assert baseline_result["passed"] is False
        # Same metrics, candidate gate → pass (streak 4 ≤ backstop 9, IC robust).
        candidate_result = eligibility(metrics_with_streak_4, PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert candidate_result["passed"] is True


# -- 4. The drawdown-conditioned streak relaxation (ADR-004 Option D) -------


class TestDrawdownConditionedStreakGate:
    """The drawdown-conditioned streak relaxation MECHANISM (ADR-004 v2).

    v3 (2026-05-29) demoted the streak to a loose OOS-stable backstop, so the
    ``portfolio_candidate`` preset no longer opts into this relaxation. The
    mechanism itself still lives in ``_streak_check`` (any threshold may set the
    DD-conditioned fields), so these tests pin it against an explicit v2-style
    threshold rather than the preset, keeping the relaxation branch covered.
    """

    # Explicit v2-style threshold: strict floor 2, relaxed cap 6 when the
    # within-streak drawdown stays inside −5%. NOT the production preset.
    _V2_THRESHOLDS = AlphaEligibilityThresholds(
        name="v2-streak-mechanism-test",
        max_fold_negative_ic_streak=2,
        max_fold_negative_ic_streak_if_dd_contained=6,
        streak_containment_max_drawdown=-0.05,
        max_drawdown=-0.10,
    )

    def _candidate_metrics(
        self,
        *,
        streak: float,
        dd_in_streak: float,
        full_dd: float = -0.04,
    ) -> dict[str, float]:
        return {
            "oos_rolling_ic": 0.10,
            "ic_60d": 0.05,
            "fold_negative_ic_streak": streak,
            "max_drawdown": full_dd,
            "max_drawdown_during_worst_streak": dd_in_streak,
            "slippage_adjusted_sharpe": 1.5,
        }

    def _streak_check(self, result: dict[str, object]) -> dict[str, object]:
        checks = result["checks"]
        assert isinstance(checks, list)
        return next(c for c in checks if c["name"] == "fold_negative_ic_streak")

    def test_streak_within_floor_passes_at_floor_threshold(self) -> None:
        # streak 2 <= floor 2: passes via the floor; the within-streak DD is
        # not even consulted (here it is deliberately catastrophic).
        metrics = self._candidate_metrics(streak=2.0, dd_in_streak=-0.99)
        check = self._streak_check(eligibility(metrics, self._V2_THRESHOLDS))
        assert check["passed"] is True
        assert check["threshold"] == 2

    def test_streak_above_floor_passes_when_dd_contained(self) -> None:
        metrics = self._candidate_metrics(streak=4.0, dd_in_streak=-0.01)
        check = self._streak_check(eligibility(metrics, self._V2_THRESHOLDS))
        assert check["passed"] is True
        assert check["threshold"] == 6  # relaxed cap applied

    def test_streak_above_floor_fails_when_episode_caused_the_drawdown(self) -> None:
        # The discriminating case: the full-run DD gate PASSES (−9% > −10%), but
        # the streak episode itself caused −8% (worse than the −5% containment
        # bound), so the relaxation is forfeit and the strict floor applies.
        metrics = self._candidate_metrics(streak=4.0, dd_in_streak=-0.08, full_dd=-0.09)
        result = eligibility(metrics, self._V2_THRESHOLDS)
        check = self._streak_check(result)
        assert check["passed"] is False
        assert check["threshold"] == 2  # reverted to floor
        # The full-run drawdown gate still passes — only the streak gate caught it.
        dd_check = next(c for c in result["checks"] if c["name"] == "max_drawdown")
        assert dd_check["passed"] is True

    def test_streak_above_hard_cap_fails_even_when_contained(self) -> None:
        # The GBDT-rank shape: streak 9 with a fully-contained episode still
        # exceeds the hard cap.
        metrics = self._candidate_metrics(streak=9.0, dd_in_streak=0.0)
        check = self._streak_check(eligibility(metrics, self._V2_THRESHOLDS))
        assert check["passed"] is False
        assert check["threshold"] == 6
