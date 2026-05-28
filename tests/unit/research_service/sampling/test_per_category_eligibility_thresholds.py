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
        # Looser streak: long-only construction absorbs negative-IC
        # stretches that would be catastrophic for an unconstrained
        # signed-rank book.
        assert (
            PORTFOLIO_CANDIDATE_THRESHOLDS.max_fold_negative_ic_streak
            > RESEARCH_RANKER_BASELINE_THRESHOLDS.max_fold_negative_ic_streak
        )

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
    """The smallest test that pins the new governance contract.

    G's actual v4 metrics (Sharpe 1.0886, DD −4.21%, streak 4)
    should:
    * FAIL the research_ranker_baseline gate (streak 4 > 2)
    * PASS the portfolio_candidate gate (every check inside the
      looser thresholds, including streak 4 == 4 and DD −4.21% >
      −10%)
    """

    def _g_metrics(self) -> dict[str, float]:
        # Verbatim from v4 universe-300 run, Arm G evidence.
        return {
            "oos_rolling_ic": 0.2561,
            "ic_60d": 0.0912,
            "fold_negative_ic_streak": 4.0,
            "max_drawdown": -0.0421,
            "slippage_adjusted_sharpe": 1.0886,
        }

    def test_g_fails_research_ranker_baseline_gate(self) -> None:
        result = eligibility(self._g_metrics(), RESEARCH_RANKER_BASELINE_THRESHOLDS)
        assert result["passed"] is False
        # The failing check should be the streak — Sharpe and DD
        # would pass even the strict baseline gate. If a future
        # tune flips this, recalibrate the test against the new
        # values; this is a regression guard, not an axiom.
        failing = [c for c in result["checks"] if not c["passed"]]
        assert len(failing) == 1
        assert failing[0]["name"] == "fold_negative_ic_streak"
        assert failing[0]["actual"] == 4
        assert failing[0]["threshold"] == 2

    def test_g_passes_portfolio_candidate_gate(self) -> None:
        result = eligibility(self._g_metrics(), PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert result["passed"] is True, (
            "G's v4 metrics MUST pass the portfolio_candidate gate; "
            "if this fails, either the threshold values changed or "
            "the gate logic regressed."
        )
        # Verify each check individually passed so a regression
        # reveals which gate slipped.
        for check in result["checks"]:
            assert check["passed"], f"check {check['name']!r} unexpectedly failed: {check}"

    def test_baseline_with_high_drawdown_still_blocked_by_dd_gate(self) -> None:
        # Defensive: a portfolio_candidate that produces a -15% DD
        # (worse than the -10% gate) is rejected even with a
        # within-bound streak. The construction-trust contract goes
        # both ways.
        broken_candidate = {
            "oos_rolling_ic": 0.10,
            "ic_60d": 0.05,
            "fold_negative_ic_streak": 1.0,
            "max_drawdown": -0.15,  # WORSE than candidate's −0.10 gate
            "slippage_adjusted_sharpe": 1.5,
        }
        result = eligibility(broken_candidate, PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert result["passed"] is False
        failing = [c for c in result["checks"] if not c["passed"]]
        # Only the DD gate should fail; everything else is loose.
        assert len(failing) == 1
        assert failing[0]["name"] == "max_drawdown"

    def test_loose_streak_does_not_help_baseline(self) -> None:
        # A research_ranker_baseline with streak=4 is rejected
        # under its own (strict) gate even though that streak would
        # pass under the portfolio_candidate gate. Categories don't
        # share thresholds — that's the whole point of the change.
        metrics_with_streak_4 = {
            "oos_rolling_ic": 0.10,
            "ic_60d": 0.05,
            "fold_negative_ic_streak": 4.0,
            "max_drawdown": -0.05,
            "slippage_adjusted_sharpe": 1.5,
        }
        baseline_result = eligibility(metrics_with_streak_4, RESEARCH_RANKER_BASELINE_THRESHOLDS)
        assert baseline_result["passed"] is False
        # Same metrics, candidate gate → pass.
        candidate_result = eligibility(metrics_with_streak_4, PORTFOLIO_CANDIDATE_THRESHOLDS)
        assert candidate_result["passed"] is True
