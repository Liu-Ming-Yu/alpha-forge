"""Full walk-forward backtest of the latest feature / strategy / alpha stack.

Fourteen arms (A-N), all on the same universe daily-bar window, identical
walk-forward grid, identical 21d log-return label, 10 bps slippage per
turnover (except Arm K, which prices a quadratic impact model), evaluated by
the existing ``run_sample_walk_forward`` driver:

* **A — research_ranker_pv**                                    : 27 price-volume
                                                                  features, signed-rank.
* **B — research_ranker_pv_formulaic**                          : + 9 formulaic
                                                                  alphas, signed-rank.
* **C — research_ranker_pv_formulaic_learnedpca**               : + 9 learned-PCA
                                                                  features, signed-rank.
* **D — long_only_top30_pv_formulaic_learnedpca**               : same 45 features,
                                                                  long-only top-30
                                                                  production-style
                                                                  constructor (5% per
                                                                  name / 22% gross /
                                                                  monthly rebal).
* **E — long_only_top30_pv_formulaic_learnedpca_streakdial**    : same as D plus a
                                                                  fold-streak exposure
                                                                  throttle (EWMA-IC +
                                                                  hard kill at 3
                                                                  consecutive negative
                                                                  folds).
* **F — long_only_top30_pv_formulaic**                          : Arm D *without*
                                                                  learned-PCA (36 features
                                                                  instead of 45).
* **G — long_only_top30_pv_formulaic_streakdial**               : Arm F + streak dial.
* **H — long_only_top30_pv_formulaic_streakdial_regime**        : Arm G + regime ×
                                                                  base-feature
                                                                  interaction overlay
                                                                  (regime-v1 family).
* **I — long_only_top30_pv_formulaic_streakdial_gbdt**          : Arm G with the linear
                                                                  IC-weighted ranker
                                                                  swapped for an XGBoost
                                                                  gradient-boosted ranker
                                                                  (MSE objective, GPU auto).
* **J — long_only_top30_pv_formulaic_streakdial_gbdt_rank**     : Arm I with a
                                                                  learning-to-rank
                                                                  objective (rank:pairwise
                                                                  + per-date query groups).
* **K — long_only_top30_pv_formulaic_streakdial_quadcost**      : Arm G priced
                                                                  through a quadratic
                                                                  market-impact cost model
                                                                  (linear 10 bps + per-name
                                                                  quadratic impact) instead
                                                                  of the flat 10 bps —
                                                                  cost-robustness arm.
* **L — long_only_top30_pv_formulaic_streakdial_invvol**        : Arm G with
                                                                  equal-weight sizing
                                                                  replaced by shrunk
                                                                  inverse-volatility
                                                                  weighting (1/vol from
                                                                  low_vol_63d, shrinkage
                                                                  0.5).
* **M — long_only_top30_pv_formulaic_streakdial_topkbuffer**    : Arm G with the
                                                                  fresh-top-N selection
                                                                  replaced by a buffered
                                                                  top-k (TopkDropout-style,
                                                                  buffer=5) to cut
                                                                  membership turnover.
* **N — long_only_top30_pv_formulaic_streakdial_gru**           : Arm G with the
                                                                  linear ranker swapped
                                                                  for a PyTorch GRU
                                                                  sequence ranker (20d
                                                                  windows, per-date IC
                                                                  loss, GPU auto).

D/E/F/G form a 2x2 ablation:

    +---------------+-------------+--------------+
    |               | no dial     | with dial    |
    +===============+=============+==============+
    | with PCA      | D           | E            |
    +---------------+-------------+--------------+
    | no  PCA       | **F**       | **G**        |
    +---------------+-------------+--------------+

D-vs-F and E-vs-G isolate **learned-PCA's contribution to the
production-candidate path** (the signed-rank A→B→C result already
shows learned-PCA hurts the ranker by Sharpe — does the same pattern
hold for long-only?). D-vs-E and F-vs-G isolate **the fold-streak
dial's contribution** (v3 confirmed +8.8% Sharpe on D→E; does it
generalise to the F shape?).

A/B/C carry ``category = research_ranker_baseline`` and
``production_candidate = False`` in their emitted evidence. Their
signed-rank weights have no per-name cap, sector neutralization, ADV cap,
borrow model, or cash model — they are diagnostic baselines that measure
whether the feature stack ranks future returns, not production
portfolios. D, E, F, and G all carry ``category = portfolio_candidate``
and ``production_candidate = True``.

Eligibility thresholds are now category-aware (ADR-003 follow-up
shipped 2026-05-28). Baselines get the strict
``RESEARCH_RANKER_BASELINE_THRESHOLDS`` (streak ≤ 2, DD ≥ −20%);
portfolio candidates get ``PORTFOLIO_CANDIDATE_THRESHOLDS`` (streak
≤ 4, DD ≥ −10%) — looser streak in exchange for tighter DD, on the
contract "we trust the construction iff it actually protects you."
The dispatch looks up ``THRESHOLDS_BY_ARM_CATEGORY[spec.category]``;
each evidence file's ``eligibility_thresholds.name`` field
identifies which set was applied.

Return accounting
-----------------
``forward_return`` is the 21-trading-day forward log return used for IC,
feature weighting, and bootstrap IC — never as a daily realized P&L.
``realized_return_1d`` is the one-day simple realized return
``close[t+1] / close[t] - 1.0`` and is the canonical input to Sharpe,
max drawdown, total return, and the equity curve. The evaluator
compounds daily returns via ``equity *= 1 + r`` and annualizes Sharpe
with ``sqrt(252)``. A side-by-side diagnostic ``metrics_bucket_21d`` is
derived by compounding the daily MtM stream into non-overlapping
21-trading-day bucket returns and annualizing with ``sqrt(252 / 21)``;
it is reported but does not feed the eligibility gate.

PIT discipline
--------------
* Feature compute is per-instrument-and-bar; no cross-sample peeking.
* The PCA artifact is fit **once** on the first 252-trading-day warmup
  window and held fixed for every test fold thereafter — mildly stale
  for late folds but cannot see future-return information.
* 21d forward log return is computed strictly forward from
  ``close[t]`` → ``close[t+21]``. ``merge_asof`` is not used.
* The walk-forward purge is set to ``purge_days = HORIZON_DAYS`` and
  ``WalkForwardConfig.label_horizon_days = HORIZON_DAYS`` so a training
  label cannot leak past the purge gap. A sample-level purge driven by
  per-row global-calendar indices (``as_of_index`` /
  ``label_end_index``) further drops train rows whose label window
  reaches the test window — stricter than the calendar gap because
  21 calendar days < 21 trading days.

Runtime
-------
Scales linearly with universe × days × arms. The first time the script
runs it computes the full feature panel; subsequent runs hit the OS
file cache.

Usage::

    python scripts/backtest_latest_stack.py
    python scripts/backtest_latest_stack.py --instrument-limit 30 --max-years 2  # smoke
    python scripts/backtest_latest_stack.py --arms A,B          # CLI aliases
    python scripts/backtest_latest_stack.py --arms research_ranker_pv  # canonical names

Output
------
JSON evidence files under
``data/parquet/research/backtest_latest_stack_realized_v2/arm_{canonical_name}.json``
plus a console comparison table. The schema version stamped on each
file is ``backtest-latest-stack-realized-v2``. Earlier directories
are preserved as frozen archives:

* ``backtest_latest_stack_v1/`` — pre-fix evidence with the
  21d-label-as-daily-P&L bug; do NOT cite.
* ``backtest_latest_stack_realized_v1/`` — v2.1 and v3 evidence
  (return-accounting fixed, label-end-index hardened, fold-streak
  dial added). Frozen 2026-05-28 when the schema bumped to v2.

The v1 → v2 root bump captures the cumulative sample-construction
changes shipped between PR #62 and PR #65 (label-end-index now
derived from the actual instrument-local future date, realized 1d
span validated against the global trading-day calendar) plus the
additive ``fold_streak_risk`` block in ``portfolio_diagnostics``.
None of those changed the *evidence consumer* contract — the v2 JSON
is a strict superset of the v1 fields — but the sampling semantics
are not bit-identical to the original v1 release.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from quant_platform.research.features.formulaic.config import (
    FEATURE_SET_VERSION as FORMULAIC_FEATURE_SET_VERSION,
)
from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.formulaic.library import LIBRARY as FORMULAIC_LIBRARY
from quant_platform.research.features.formulaic.panel import build_market_panel
from quant_platform.research.features.learned.artifact import (
    ARTIFACT_SCHEMA_VERSION as LEARNED_PCA_ARTIFACT_SCHEMA_VERSION,
)
from quant_platform.research.features.learned.config import (
    DEFAULT_CONFIG as LEARNED_CONFIG,
)
from quant_platform.research.features.learned.config import (
    FEATURE_SET_VERSION as LEARNED_FEATURE_SET_VERSION,
)
from quant_platform.research.features.learned.features import compute_learned_features
from quant_platform.research.features.learned.trainer import fit_pca_artifact
from quant_platform.research.features.price_volume.config import (
    FEATURE_SET_VERSION as PV_FEATURE_SET_VERSION,
)
from quant_platform.research.features.price_volume.features import (
    compute_price_volume_features,
)
from quant_platform.research.features.regime.config import (
    FEATURE_SET_VERSION as REGIME_FEATURE_SET_VERSION,
)
from quant_platform.research.features.regime.features import (
    compute_regime_features,
    regime_detector_metadata,
)
from quant_platform.services.research_service.campaigns.evaluation.artifacts import (
    current_git_commit,
)
from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.metrics.return_metrics import (
    bucket_sharpe,
    compound_return,
    max_drawdown,
    non_overlapping_bucket_returns,
)
from quant_platform.services.research_service.campaigns.models.gbdt import GradientBoostedRanker
from quant_platform.services.research_service.campaigns.models.robust_linear import RobustICRanker
from quant_platform.services.research_service.campaigns.models.sequence import GRUSequenceRanker
from quant_platform.services.research_service.campaigns.portfolio.costs import QuadraticImpactCost
from quant_platform.services.research_service.campaigns.portfolio.selection import (
    BufferedTopKSelection,
)
from quant_platform.services.research_service.campaigns.portfolio.streak_risk import (
    FoldStreakRiskConfig,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
)
from quant_platform.services.research_service.campaigns.portfolio.weighting import (
    ConvictionWeight,
    InverseVolWeight,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.factory_models import (
    THRESHOLDS_BY_ARM_CATEGORY,
    AlphaEligibilityThresholds,
    WalkForwardEvidence,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from quant_platform.services.research_service.campaigns.models.base import AlphaModel
    from quant_platform.services.research_service.campaigns.portfolio.costs import TradingCostModel
    from quant_platform.services.research_service.campaigns.portfolio.selection import (
        SelectionStrategy,
    )
    from quant_platform.services.research_service.campaigns.portfolio.weighting import (
        WeightingScheme,
    )
    from quant_platform.services.research_service.sampling.arm_category import ArmCategory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAR_ROOT = PROJECT_ROOT / "data" / "parquet" / "bars"
UNIVERSE_FILE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "parquet" / "research" / "backtest_latest_stack_realized_v2"
#: Evidence schema version. Minor revisions (``v2.0`` → ``v2.1`` …)
#: signal **additive** changes only — readers iterating by key still
#: work, readers expecting strict-dict equality across runs need to
#: tolerate new keys. Major revisions (``v2`` → ``v3``) signal
#: breaking changes (renamed/removed keys, semantic shifts) and
#: trigger an OUTPUT_ROOT bump so old and new evidence don't co-mingle.
#:
#: ``v2.1`` (PR #71): added ``eligibility_thresholds.name`` field.
#: ``v2`` (PR #66): initial realized-accounting schema.
EVIDENCE_SCHEMA_VERSION = "backtest-latest-stack-realized-v2.1"

# HORIZON_DAYS matches ``TRADING_DAYS_PER_MONTH`` (21) by design — a
# 21-trading-day forward return is "1 month" in the project's calendar
# token system. Kept as a local constant rather than importing the
# token so this script stays self-contained as a research entry point;
# if you change the horizon, also revisit anywhere
# TRADING_DAYS_PER_MONTH-based features are tuned (price_volume
# momentum lookbacks, formulaic alphas with ``ts_*`` windows).
HORIZON_DAYS = 21
# 504d = 2 years: the longest price-volume lookbacks (ret_252d, 52w-high,
# 252d drawdown) need 253 bars before their first non-NaN row. Setting
# the warmup to 504 gives ~250 days × universe-size clean rows for the
# PCA fit while still leaving the bulk of history for OOS folds.
WARMUP_TRADING_DAYS = 504
SLIPPAGE_BPS = 10.0
FEATURE_SET_VERSION = "latest-stack-v1"
MODEL_VERSION = "ic-weighted-non-negative"


# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------


# ``ArmCategory`` Literal lives in the shared
# ``services.research_service.sampling.arm_category`` module so the
# eligibility-threshold lookup in ``factory_models`` can type itself
# against it directly; see ADR-004 for the framing. It is re-exported
# from ``factory_models`` for convenience.

#: Which prebuilt feature panel an arm consumes. Each value maps to a
#: ``(panel_df, feature_names, feature_set_versions)`` triple in the
#: dispatch loop's panel registry.
#:
#: ``pv_form_regime`` = ``pv_form`` augmented with regime interaction
#: features (regime-v1 family, see ADR-005). The regime panel is only
#: built when at least one requested arm declares
#: ``panel_key="pv_form_regime"``.
#:
#: ``pv_form_fund`` = ``pv_form`` augmented with the 9 Sharadar quality+value
#: fundamentals (PIT-aligned), an alpha source orthogonal to momentum — tests
#: whether it diversifies the momentum-crash fragility. Built only when a
#: requested arm declares ``panel_key="pv_form_fund"``.
PanelKey = Literal["pv", "pv_form", "full", "pv_form_regime", "pv_form_fund"]


@dataclass(frozen=True)
class ArmSpec:
    """Declarative description of one backtest arm.

    Adding a new arm is a single-tuple addition to ``ARM_SPECS`` plus
    (for portfolio arms) a ``portfolio_config_factory`` callable and
    optionally a ``fold_streak_risk_config_factory`` (for portfolio
    arms that opt into the ADR-pending fold-streak exposure throttle).
    ``panel_key`` chooses one of the prebuilt feature panels; the
    dispatch loop does the rest uniformly.
    """

    cli_alias: str
    canonical_name: str
    category: ArmCategory
    production_candidate: bool
    panel_key: PanelKey
    requires_pca: bool = False
    portfolio_config_factory: Callable[[], CampaignPortfolioConfig] | None = None
    fold_streak_risk_config_factory: Callable[[], FoldStreakRiskConfig] | None = None
    #: Optional pluggable alpha model. ``None`` keeps the default linear
    #: IC-weighted ranker (arms A-H). Must be a module-level callable so the
    #: spec stays picklable for the ProcessPoolExecutor (Windows spawn); the
    #: model itself is constructed inside the worker, never pickled.
    model_factory: Callable[[], AlphaModel] | None = None
    #: Optional pluggable trading-cost model (long-only arms only). ``None``
    #: keeps the behavior-preserving flat 10 bps/turnover linear cost (arms
    #: A-J). Arm K supplies a quadratic impact model. Same picklability rule as
    #: ``model_factory``: a module-level callable, built inside the worker.
    cost_model_factory: Callable[[], TradingCostModel] | None = None
    #: Optional pluggable position-weighting scheme (long-only arms only).
    #: ``None`` keeps equal-weight (arms A-K). Arm L supplies inverse-vol
    #: weighting. Same picklability rule as ``model_factory``.
    weighting_factory: Callable[[], WeightingScheme] | None = None
    #: Optional pluggable name-selection strategy (long-only arms only). ``None``
    #: keeps the fresh top-N (arms A-L). Arm M supplies a buffered-top-k
    #: (TopkDropout-style) selection. Same picklability rule as ``model_factory``.
    selection_factory: Callable[[], SelectionStrategy] | None = None


def _long_only_top30_config() -> CampaignPortfolioConfig:
    return CampaignPortfolioConfig(
        mode="runtime-long-only",
        top_n=30,
        vol_target=0.15,
        vol_floor=0.05,
        vol_lookback_days=63,
        max_gross_exposure=0.22,
        min_cash_buffer=0.05,
        max_single_name_weight=0.05,
        max_daily_turnover=0.20,
        max_position_change=0.05,
        no_trade_band=0.005,
        rebalance_interval_days=21,
    )


def _default_fold_streak_risk_config() -> FoldStreakRiskConfig:
    """Starting-point thresholds for the fold-streak exposure throttle.

    Tuned against the audit-binding ``fold_negative_ic_streak <= 2``
    gate: ``kill_streak=3`` triggers the circuit breaker on the first
    streak that would breach the gate. EWMA halflife of 4 folds
    smooths over ~84 calendar days of OOS evidence. Floor/ceiling
    framed around zero so the throttle disengages whenever recent
    OOS IC is non-negative on average.
    """
    return FoldStreakRiskConfig(
        min_folds_before_active=4,
        kill_streak=3,
        ewma_halflife=4,
        floor_ic=-0.02,
        ceiling_ic=0.0,
    )


def _gbdt_ranker_factory() -> AlphaModel:
    """XGBoost gradient-boosted ranker for Arm I (GPU auto-detect, CPU fallback).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. The booster is built per fold inside
    the worker; ``device="auto"`` uses CUDA when a working build is present
    (e.g. the RTX 5080 here) and silently falls back to CPU otherwise.
    """
    return GradientBoostedRanker(objective="regression", device="auto")


def _gbdt_rank_ranker_factory() -> AlphaModel:
    """XGBoost learning-to-rank ranker for Arm J (rank:pairwise, GPU auto).

    Same as Arm I's factory but with a pairwise-ranking objective and per-date
    query groups. Targets the IC-quality gap that the MSE objective (Arm I) left
    open: ranking the cross-section directly rather than predicting return levels.
    """
    return GradientBoostedRanker(objective="rank", device="auto")


def _gru_ranker_factory() -> AlphaModel:
    """PyTorch GRU sequence ranker for Arm N (IC loss, GPU auto-detect, CPU fallback).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. The net is built per fold inside the
    worker; ``device="auto"`` uses CUDA when available (the RTX 5080 here) and
    falls back to CPU. ``objective="ic"`` (per-date Pearson IC loss) is the right
    default given the Arm I/J lesson that MSE-on-levels ranks poorly. Requires
    the ``dl`` extra (torch); if absent the worker surfaces the arm as errored
    rather than tearing down the run.
    """
    return GRUSequenceRanker(objective="ic", device="auto")


def _robust_ic_ranker_factory() -> AlphaModel:
    """Downside-robust IC-weighted ranker for Arm O (ADR-004/011 follow-up).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. Same linear scoring as G (returns a
    ``_FittedLinearRanker`` → rank-normalized weighted sum), but the per-feature
    weights are fit from the downside of each feature's daily-IC distribution so
    crash-fragile medium-term momentum is down-weighted in favour of the
    crash-robust volatility/range, reversal, and long-horizon factors. Tests
    whether the in-sample robustness gain (IC-IR ~0.09 → ~0.21, crash IC flipped
    positive) survives out-of-sample. Defaults (downside_weight 2.0, q 0.33)
    match the feasibility sweep.
    """
    return RobustICRanker()


def _quadratic_impact_cost_factory() -> TradingCostModel:
    """Quadratic market-impact cost model for Arm K (qlib plan item 2).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. Keeps the same 10 bps/turnover linear
    spread/commission as every other arm and adds a per-name quadratic impact
    term anchored at 10 bps for a full single-name-cap (0.05) trade — the convex
    cost the flat model ignored. See ADR-007. The anchor is a documented
    modeling assumption (no per-name ADV in the weight-space evaluator), so Arm K
    is a cost-*robustness* test of G, not a venue-calibrated one.
    """
    return QuadraticImpactCost(
        linear_bps_per_turnover=SLIPPAGE_BPS,
        impact_bps_at_cap=10.0,
        single_name_cap=0.05,
    )


def _inverse_vol_weight_factory() -> WeightingScheme:
    """Shrunk inverse-volatility weighting for Arm L (qlib plan item 3).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. Sizes the top-30 selected names by
    1/vol, read from the point-in-time ``low_vol_63d`` price-volume feature (the
    63d window matches the config's ``vol_lookback_days``). ``shrinkage=0.5``
    blends the inverse-vol tilt halfway back toward equal weight — a documented
    middle ground that blunts single-estimate vol noise without abandoning the
    risk tilt. See ADR-008. Selection (the alpha) is unchanged from G; only the
    per-name sizing differs.
    """
    return InverseVolWeight(vol_feature="low_vol_63d", shrinkage=0.5, vol_floor=0.005)


def _conviction_weight_factory() -> WeightingScheme:
    """Conviction-proportional weighting for Arm Q (the IC->Sharpe / TC lever).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. Sizes the top-30 by alpha conviction
    (score above the selection floor) rather than equal weight, raising the
    transfer coefficient in the Fundamental Law (IR = IC*sqrt(BR)*TC). The
    rank-normalized composite still chooses the names; only sizing differs.
    ``shrinkage=0.25`` is the sweep-selected peak: it blends a quarter back to
    equal weight to blunt estimation error (the book's "1/N beats 14 optimizers"
    guardrail). The shrinkage sweep (2026-05-29) is decisive and non-monotonic —
    pure conviction (0.0) *underperforms* (Sharpe 0.956) because zeroing the
    marginal names over-concentrates (lower effective breadth); 0.25 peaks at
    Sharpe 1.0211 and 0.5 is ~tied at 1.018. No ``vol_feature`` — pure conviction
    tilt to isolate the transfer-coefficient lever from Arm L's risk tilt.
    """
    return ConvictionWeight(shrinkage=0.25, reference="min")


def _buffered_topk_selection_factory() -> SelectionStrategy:
    """Buffered top-k (TopkDropout-style) selection for Arm M (qlib plan item 5).

    Module-level (not a lambda) so the owning ``ArmSpec`` pickles for the
    ProcessPoolExecutor on Windows-spawn. Keeps a held name in the top-30 until
    it slips past rank ``30 + buffer`` (buffer=5 here), so a name that wiggles
    around the cutoff is not churned. Targets membership-driven turnover, the
    cost the no-trade band + position cap only partly smooth. See ADR-009.
    """
    return BufferedTopKSelection(buffer=5)


ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec(
        cli_alias="A",
        canonical_name="research_ranker_pv",
        category="research_ranker_baseline",
        production_candidate=False,
        panel_key="pv",
    ),
    ArmSpec(
        cli_alias="B",
        canonical_name="research_ranker_pv_formulaic",
        category="research_ranker_baseline",
        production_candidate=False,
        panel_key="pv_form",
    ),
    ArmSpec(
        cli_alias="C",
        canonical_name="research_ranker_pv_formulaic_learnedpca",
        category="research_ranker_baseline",
        production_candidate=False,
        panel_key="full",
        requires_pca=True,
    ),
    ArmSpec(
        cli_alias="D",
        canonical_name="long_only_top30_pv_formulaic_learnedpca",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="full",
        requires_pca=True,
        portfolio_config_factory=_long_only_top30_config,
    ),
    # Arm E = Arm D + fold-streak exposure throttle. Direct A/B test of
    # whether cutting exposure during negative-IC streaks recovers the
    # eligibility gate that D fails (fold_negative_ic_streak = 7 vs ≤ 2).
    # Same panel, same long-only construction, same portfolio config —
    # only the streak dial differs.
    ArmSpec(
        cli_alias="E",
        canonical_name="long_only_top30_pv_formulaic_learnedpca_streakdial",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="full",
        requires_pca=True,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
    ),
    # Arm F = Arm D without learned-PCA. Long-only top-30, same
    # portfolio config, same 21-day rebalance — but only the 27 PV
    # features + 9 formulaic alphas, no PCA. Together with G this
    # completes the 2x2 ablation:
    #
    #              | no dial | with dial |
    #   ----------+---------+-----------+
    #   with PCA  |   D     |    E      |
    #   no  PCA   |  *F*    |   *G*     |
    #
    # D vs F and E vs G isolate learned-PCA's contribution; D vs E and
    # F vs G isolate the dial's contribution. The signed-rank arms
    # already show C < B on Sharpe (learned-PCA hurts the ranker);
    # F+G tests whether the same pattern holds on the production-
    # candidate long-only path.
    ArmSpec(
        cli_alias="F",
        canonical_name="long_only_top30_pv_formulaic",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
    ),
    # Arm G = Arm F + fold-streak exposure throttle. Mirror of D vs E
    # one ablation axis over: same dial as E but consuming the
    # no-PCA panel. Confirms (or contradicts) D-vs-E's "the dial
    # gives ~+8.8% Sharpe" finding on the F shape.
    ArmSpec(
        cli_alias="G",
        canonical_name="long_only_top30_pv_formulaic_streakdial",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
    ),
    # Arm H = Arm G + regime overlay (ADR-005). Same long-only
    # construction + streak dial as G, but the feature set is
    # augmented with regime × base-feature interactions from the
    # ``regime-v1`` family. Direct A/B test: does giving the
    # IC-weighted ranker regime-conditioned features move the
    # ``fold_negative_ic_streak`` metric? G passes eligibility at
    # streak=4 (gate ≤ 4) with zero margin; H either widens the
    # margin (regime overlay works) or doesn't (curation needs
    # rework, or Shape C is required).
    ArmSpec(
        cli_alias="H",
        canonical_name="long_only_top30_pv_formulaic_streakdial_regime",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form_regime",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
    ),
    # Arm I = Arm G's exact construction (no-PCA pv+formulaic panel, long-only
    # top-30, streak dial) with the linear IC-weighted ranker swapped for an
    # XGBoost gradient-boosted ranker. Direct A/B vs the production lead G:
    # does a nonlinear learner that captures feature *interactions* move the
    # binding ``fold_negative_ic_streak`` gate that the linear ranker, the
    # dial, and the regime overlay could not? Same 36 features, same portfolio
    # construction, same dial — only the model differs, so any delta is the
    # model's. Requires the ``ml`` extra (xgboost); if absent the worker
    # surfaces the arm as errored rather than tearing down the run.
    ArmSpec(
        cli_alias="I",
        canonical_name="long_only_top30_pv_formulaic_streakdial_gbdt",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        model_factory=_gbdt_ranker_factory,
    ),
    # Arm J = Arm I with a learning-to-rank objective (rank:pairwise + per-date
    # query groups) instead of MSE. Arm I showed the GBDT picks winners (higher
    # total return) but ranks the cross-section poorly (ic_60d 0.028 < 0.03 gate,
    # decile spread collapses) — MSE-on-levels is the wrong loss for a ranker.
    # J optimizes the ordering the IC gate measures directly. Same panel, same
    # construction, same dial — only the objective differs. Expectation: IC
    # quality recovers; if the streak still holds at 4 it confirms the streak is
    # model-invariant on this universe (ADR-006).
    ArmSpec(
        cli_alias="J",
        canonical_name="long_only_top30_pv_formulaic_streakdial_gbdt_rank",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        model_factory=_gbdt_rank_ranker_factory,
    ),
    # Arm K = Arm G's exact construction (linear ranker, no-PCA pv+formulaic,
    # long-only top-30, streak dial) priced through a quadratic market-impact
    # cost model instead of the flat 10 bps/turnover. Only the cost accounting
    # differs — the alpha, ranking, and weights are identical to G, so IC and
    # streak are unchanged by construction; what moves is the post-cost Sharpe
    # and return. This is the cost-robustness arm: does G's eligibility survive
    # a convex impact model? It directly answers the pre-paper hardening concern
    # that real slippage may be materially worse than the flat assumption. See
    # ADR-007 (qlib plan item 2).
    ArmSpec(
        cli_alias="K",
        canonical_name="long_only_top30_pv_formulaic_streakdial_quadcost",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        cost_model_factory=_quadratic_impact_cost_factory,
    ),
    # Arm L = Arm G's exact construction (linear ranker, no-PCA pv+formulaic,
    # long-only top-30, streak dial, flat 10 bps cost) with equal-weight sizing
    # replaced by shrunk inverse-volatility weighting. Only the per-name SIZE
    # distribution differs — selection (the alpha) and gross budget are
    # identical to G, so the selected set is unchanged; what moves is risk
    # concentration and therefore Sharpe/turnover. Does sizing by 1/vol lift
    # risk-adjusted return over equal weight on this universe? See ADR-008
    # (qlib plan item 3).
    ArmSpec(
        cli_alias="L",
        canonical_name="long_only_top30_pv_formulaic_streakdial_invvol",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        weighting_factory=_inverse_vol_weight_factory,
    ),
    # Arm M = Arm G's exact construction (linear ranker, no-PCA pv+formulaic,
    # long-only top-30, streak dial, flat 10 bps cost, equal weight) with the
    # fresh-top-N selection replaced by a buffered top-k (TopkDropout-style):
    # a held name keeps its slot until it slips past rank 30 + buffer, so a
    # name wiggling around the cutoff isn't churned. Changes the held book
    # (returns + turnover) but not the IC, which the driver measures over the
    # full scored cross-section, not the held names. Does cutting
    # membership-driven turnover beat the cost of holding slightly-lower-ranked
    # names on a construction that already has a no-trade band + caps + dial?
    # See ADR-009 (qlib plan item 5).
    ArmSpec(
        cli_alias="M",
        canonical_name="long_only_top30_pv_formulaic_streakdial_topkbuffer",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        selection_factory=_buffered_topk_selection_factory,
    ),
    # Arm N = Arm G's exact construction (no-PCA pv+formulaic panel, long-only
    # top-30, streak dial) with the linear IC ranker swapped for a PyTorch GRU
    # sequence ranker (per-date IC loss, GPU auto). The GRU consumes a 20-day
    # sequence of each name's recent feature vectors (reconstructed via
    # as_of_index + a train-tail cache, with a log1p time-gap channel) rather
    # than a single row. The qlib model-zoo sequence learner ADR-006 deferred;
    # proves the AlphaModel seam carries a heavy torch model. Same 36 features,
    # same construction — only the model differs. Honest prior: unlikely to beat
    # the linear ranker's IC at this horizon, so most likely an informative
    # negative result. Requires the `dl` extra (torch); absent -> worker errors
    # the arm without tearing down the run. See ADR-010.
    ArmSpec(
        cli_alias="N",
        canonical_name="long_only_top30_pv_formulaic_streakdial_gru",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        model_factory=_gru_ranker_factory,
    ),
    # Arm O = Arm G's exact construction (linear-family, no-PCA pv+formulaic,
    # long-only top-30, streak dial) but with the downside-robust IC ranker
    # instead of mean-IC weighting. The regime decomposition showed G's
    # medium-term momentum inverts during crashes; O fits weights from the
    # downside of each feature's daily-IC distribution to drop crash-fragile
    # factors and keep crash-robust ones (vol/range, reversal, long horizon).
    # OOS VERDICT (2026-05-29, normfixed universe-300): NEGATIVE. The in-sample
    # robustness gain (IC-IR 0.09→0.21) did NOT survive the walk-forward. O cuts
    # the streak (G 7 → O 4 — the crash-robust features do hold) but GUTS the
    # alpha: oos_ic 0.159→0.041, ic_60d 0.063→−0.029, bootstrap_p05 +0.015→
    # −0.015, decile spread +0.004→−0.003; Sharpe 0.852→0.816. The crash-robust
    # features are defensive but weak — dropping momentum removes the fragility
    # AND the alpha. Confirms the alpha is intrinsically momentum (can't keep it
    # without the crash exposure via a linear reweighting). Kept as a documented
    # negative result in the research ledger. O stays a portable LINEAR model.
    ArmSpec(
        cli_alias="O",
        canonical_name="long_only_top30_pv_formulaic_streakdial_robustic",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        model_factory=_robust_ic_ranker_factory,
    ),
    # Arm P = Arm G's construction (linear ranker, long-only top-30, streak
    # dial) on the pv+formulaic panel AUGMENTED with the 9 Sharadar quality+
    # value fundamentals (PIT-aligned). Fundamentals are orthogonal to momentum;
    # hypothesis: they diversify G's crash fragility and lift risk-adjusted
    # return. Same model/construction/dial as G; only the panel differs.
    # OOS VERDICT (2026-05-29, normfixed universe-300): PARTIAL / positive on IC,
    # insufficient for eligibility. Fundamentals ADD genuine orthogonal alpha —
    # oos_ic 0.159→0.190 (+20%), ic_60d 0.063→0.072, bootstrap_p05 0.0152→0.0169,
    # decile spread 0.0045→0.0058 (all up, no harm) — but do NOT move the binding
    # gates: Sharpe 0.852→0.838 (still <1.0) and streak unchanged at 7. The
    # orthogonal signal improves *average* ranking but doesn't fix the momentum-
    # crash *timing* (streak) or translate the IC gain into Sharpe. Kept as a
    # real (if not gate-passing) improvement; confirms IC is improvable while
    # Sharpe stays sticky (the IC→Sharpe translation is now the binding lever).
    ArmSpec(
        cli_alias="P",
        canonical_name="long_only_top30_pv_formulaic_fundamentals_streakdial",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form_fund",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
    ),
    # Arm Q = Arm G + CONVICTION weighting (the IC->Sharpe / transfer-coefficient
    # lever, from "Quantitative Trading Strategies" Ch.12 + the Fundamental Law
    # IR=IC*sqrt(BR)*TC). Same alpha/selection/dial/panel as G; only the per-name
    # sizing differs: top-30 are sized by alpha conviction (score above the
    # selection floor, shrunk 0.25 toward equal weight) instead of equal weight.
    # OOS VERDICT (2026-05-29, normfixed universe-300): BREAKTHROUGH. Raising TC
    # converts G's diffuse IC into Sharpe exactly as the Fundamental Law predicts.
    # IC is UNCHANGED (selection identical → oos_ic 0.1586, ic_60d 0.063,
    # bootstrap_p05 0.0152, streak 7 all identical to G), but Sharpe 0.852→1.0211
    # (shrinkage-0.25, the sweep peak; CROSSES the 1.0 gate) and total_return
    # 0.124→0.156 (+26%), DD ~−0.036, turnover ~flat. Q PASSES the v3 gate
    # (oos_ic/ic_60d/streak≤9/DD/Sharpe/bootstrap_p05 all clear) — the FIRST
    # portable LINEAR arm to do so (deploys via the existing pv_formulaic live
    # port + ConvictionWeight, no PCA/GRU). The PRODUCTION LEAD. Thin margin
    # (1.021); the cost framework (per-name √-impact + no-trade band) is the
    # planned hardening. Distinct from Arm L (inverse-vol): L re-shaped RISK with
    # no conviction and failed; Q adds conviction — that is the whole difference.
    ArmSpec(
        cli_alias="Q",
        canonical_name="long_only_top30_pv_formulaic_streakdial_conviction",
        category="portfolio_candidate",
        production_candidate=True,
        panel_key="pv_form",
        requires_pca=False,
        portfolio_config_factory=_long_only_top30_config,
        fold_streak_risk_config_factory=_default_fold_streak_risk_config,
        weighting_factory=_conviction_weight_factory,
    ),
)
ARM_SPEC_BY_KEY: dict[str, ArmSpec] = {}
for _spec in ARM_SPECS:
    ARM_SPEC_BY_KEY[_spec.cli_alias] = _spec
    ARM_SPEC_BY_KEY[_spec.canonical_name] = _spec
# Uniqueness invariant: each spec contributes both its CLI alias and
# canonical name, so the key count must be exactly 2× the spec count.
# Two specs with the same alias would silently overwrite each other.
# Use ``raise`` rather than ``assert`` so the check survives ``python -O``
# (assertions are stripped under optimisation).
if len(ARM_SPEC_BY_KEY) != 2 * len(ARM_SPECS):
    raise RuntimeError(
        f"duplicate ArmSpec identifier(s) in ARM_SPECS: "
        f"got {len(ARM_SPEC_BY_KEY)} keys for {len(ARM_SPECS)} specs"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_universe() -> dict[str, dict[str, object]]:
    with UNIVERSE_FILE.open() as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if v.get("active", True)}


def load_bars(
    instrument_ids: Iterable[str],
    *,
    max_years: int | None = None,
) -> pd.DataFrame:
    """Load every instrument's daily bars from the local parquet store."""
    frames: list[pd.DataFrame] = []
    instrument_list = list(instrument_ids)
    n_total = len(instrument_list)
    cutoff_year = None
    if max_years is not None:
        cutoff_year = 2026 - max_years + 1
    for i, inst in enumerate(instrument_list, 1):
        daily = BAR_ROOT / inst / "daily"
        files = sorted(daily.glob("*.parquet"))
        if not files:
            continue
        for f in files:
            if cutoff_year is not None:
                try:
                    year = int(f.stem)
                except ValueError:
                    continue
                if year < cutoff_year:
                    continue
            df = pd.read_parquet(
                f,
                columns=["timestamp", "bar_seconds", "open", "high", "low", "close", "volume"],
            )
            df = df[df["bar_seconds"] == 86400]
            if df.empty:
                continue
            df = df.copy()
            df["instrument_id"] = inst
            df["date"] = (
                pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.normalize()
            )
            frames.append(df[["instrument_id", "date", "open", "high", "low", "close", "volume"]])
        if i % 50 == 0:
            print(f"    loaded {i}/{n_total} instruments", flush=True)
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.sort_values(["instrument_id", "date"]).drop_duplicates(
        subset=["instrument_id", "date"]
    )
    bars["date"] = bars["date"].dt.tz_localize(None)  # naive for downstream math
    return bars.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_pv_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ff = compute_price_volume_features(bars)
    return ff.frame, list(ff.feature_names)


def compute_formulaic_alphas(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    panel = build_market_panel(bars)
    cache = ExpressionCache()
    cols: dict[str, np.ndarray] = {}
    for entry in FORMULAIC_LIBRARY:
        series = evaluate_expression(panel, entry.expression, cache=cache).astype(float)
        cols[entry.name] = series.to_numpy()
    out = pd.DataFrame(
        {
            "instrument_id": panel.frame["instrument_id"].to_numpy(),
            "date": panel.frame["date"].to_numpy(),
            **cols,
        }
    )
    return out, list(cols.keys())


#: Evidence label for the Sharadar quality+value fundamentals panel (Arm P).
FUND_FEATURE_SET_VERSION = "sharadar-quality-value-v1"


def compute_fundamentals_panel(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """PIT quality+value fundamentals panel on the daily bar grid (Arm P).

    Loads Sharadar SF1, computes the 9 quality+value features
    (``research.fundamentals.compute_starter_features``), and per instrument
    forward-fills the most-recent feature row whose ``datekey`` is strictly
    before the trading day (tradable the session *after* the filing) via
    ``merge_asof``. Returns ``(panel_df, feature_names)`` keyed
    ``(instrument_id, date)`` to match the pv/formulaic panels. Instruments or
    dates without a known filing carry ``NaN`` — the rank-normalizing scorer maps
    those to the neutral median, so a name with no fundamentals is simply ranked
    on its pv+formulaic features.
    """
    from quant_platform.research.fundamentals import (  # noqa: PLC0415
        FEATURE_NAMES as _FUND_NAMES,
    )
    from quant_platform.research.fundamentals import (  # noqa: PLC0415
        compute_starter_features,
        load_sharadar_sf1_panel,
    )

    fund_names = list(_FUND_NAMES)
    sf1 = load_sharadar_sf1_panel()
    ff = compute_starter_features(sf1).frame.copy()
    ff["instrument_id"] = ff["instrument_id"].astype(str)
    datekey = pd.to_datetime(ff["datekey"])
    if getattr(datekey.dt, "tz", None) is not None:
        datekey = datekey.dt.tz_localize(None)
    # Match the bars' datetime resolution so merge_asof keys are compatible
    # (parquet bar timestamps load as datetime64[us]; pd.to_datetime is [ns]).
    ff["datekey"] = datekey.astype("datetime64[ns]")
    ff = ff[["instrument_id", "datekey", *fund_names]].sort_values("datekey")

    grid = bars[["instrument_id", "date"]].copy()
    grid["instrument_id"] = grid["instrument_id"].astype(str)
    grid["date"] = grid["date"].astype("datetime64[ns]")
    frames: list[pd.DataFrame] = []
    for instrument_id, group in grid.groupby("instrument_id", sort=False):
        per_inst = ff[ff["instrument_id"] == instrument_id]
        if per_inst.empty:
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            per_inst.drop(columns="instrument_id"),
            left_on="date",
            right_on="datekey",
            direction="backward",
            allow_exact_matches=False,  # tradable the session AFTER the filing
        )
        merged["instrument_id"] = instrument_id
        frames.append(merged[["instrument_id", "date", *fund_names]])

    if not frames:
        empty = pd.DataFrame(columns=["instrument_id", "date", *fund_names])
        return empty, fund_names
    return pd.concat(frames, ignore_index=True), fund_names


def fit_warmup_pca_artifact(
    combined: pd.DataFrame,
    source_feature_names: list[str],
    warmup_end_date: pd.Timestamp,
) -> object:
    warmup = combined[combined["date"] <= warmup_end_date].copy()
    warmup[source_feature_names] = warmup[source_feature_names].replace([np.inf, -np.inf], np.nan)
    n_warmup_rows = len(warmup)
    n_clean = int(warmup[source_feature_names].dropna().shape[0])
    print(
        f"      warmup window: {n_warmup_rows:,} rows total, "
        f"{n_clean:,} clean rows after NaN+inf drop"
    )
    if n_clean < 250:
        raise ValueError(
            f"insufficient clean rows in warmup window: {n_clean} "
            f"(need >= 250 for n_components=8 PCA). Try a longer "
            f"warmup or a smaller feature subset."
        )
    return fit_pca_artifact(
        panel=warmup,
        feature_names=source_feature_names,
        n_components=LEARNED_CONFIG.expected_n_components,
        family_version=LEARNED_CONFIG.version,
        drop_nan_rows=True,
        validate_against_registry=False,
        extra_metadata={
            "warmup_end_date": warmup_end_date.isoformat(),
            "warmup_clean_rows": str(n_clean),
        },
    )


def compute_learned_features_panel(
    combined: pd.DataFrame,
    source_feature_names: list[str],
    artifact: object,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the deterministic compute path with the warmup artifact."""
    # Sanitize ±inf -> NaN so the compute path masks them out cleanly.
    clean = combined.copy()
    clean[source_feature_names] = clean[source_feature_names].replace([np.inf, -np.inf], np.nan)
    ff = compute_learned_features(panel=clean, artifact=artifact, config=LEARNED_CONFIG)  # type: ignore[arg-type]
    return ff.frame, list(ff.feature_names)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def build_supervised_samples(
    feature_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    feature_names: list[str],
    sector_map: dict[str, str],
    *,
    horizon_days: int = HORIZON_DAYS,
    global_calendar: pd.DatetimeIndex | None = None,
) -> list[SupervisedAlphaSample]:
    """Build supervised samples with the 21d label and a 1d realized return.

    Each sample carries:

    * ``forward_return`` — ``log(close[t+horizon] / close[t])`` where the
      ``+horizon`` is an **instrument-local** row shift (the 21st
      subsequent bar that instrument actually has). Used as the
      predictive label for IC, weighting, and bootstrap CIs.
    * ``realized_return_1d`` — ``close[next_bar] / close[t] - 1.0``,
      a simple return suitable for the evaluator's ``equity *= 1 + r``
      compounding. Only emitted when ``next_bar`` is exactly the next
      *global* trading day; rows where the instrument skipped a day
      are dropped so the "one-day close-to-close" contract holds.
    * ``as_of_index`` / ``label_end_index`` — integer positions on the
      *global* sorted trading-day calendar. ``label_end_index`` is the
      position of the **actual** instrument-local label-end date, not
      ``as_of_index + horizon`` — those differ whenever an instrument
      has missing bars (halts, late starts, data gaps). The sample-level
      purge in ``run_sample_walk_forward`` relies on this being the
      truth: a calendar-offset shortcut would let a sample whose real
      label reaches deep into the test window slip past the purge.
    * ``label_end_as_of`` — datetime version of the same instrument-
      local label-end date.

    Rows are dropped when: features are entirely non-finite, forward
    return is non-finite, realized 1d return is non-finite, the
    instrument-local label-end date can't be located on the global
    calendar, or the next instrument bar is not the next global
    trading day. There is no quiet fallback to a partial label.
    """
    samples: list[SupervisedAlphaSample] = []
    merged = feature_panel.merge(close_panel, on=["instrument_id", "date"], how="left")
    merged = merged.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # Global sorted trading-day calendar across all instruments. Per-row
    # indices are computed against this calendar so the sample-level purge
    # in run_sample_walk_forward can compare label spans consistently —
    # per-instrument row numbers would misalign when instruments have
    # missing bars.
    if global_calendar is None:
        global_calendar = pd.DatetimeIndex(pd.Series(merged["date"].unique()).sort_values())

    for inst, group in merged.groupby("instrument_id", sort=False):
        if len(group) <= horizon_days:
            continue
        group = group.sort_values("date").reset_index(drop=True)
        close = group["close"].to_numpy(dtype=float)

        # Instrument-local forward log return — the IC label. The shift
        # is over THIS INSTRUMENT'S rows, not over the global calendar,
        # so on an instrument with gaps the label-end date can be
        # significantly later than ``as_of + horizon`` calendar days.
        future_close = np.empty_like(close)
        future_close[:-horizon_days] = close[horizon_days:]
        future_close[-horizon_days:] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.log(future_close / close)

        # 1d realized SIMPLE return — the canonical P&L unit. Simple
        # because the evaluator compounds via ``equity *= 1 + r``; a log
        # return there would be a subtler version of the bug we're fixing.
        next_close = np.empty_like(close)
        next_close[:-1] = close[1:]
        next_close[-1] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            realized = next_close / close - 1.0

        # Global-calendar positions: where this row's as_of, next-bar,
        # and label-end dates land on the cross-instrument calendar.
        dates_index = pd.DatetimeIndex(group["date"])
        as_of_indices = global_calendar.get_indexer(dates_index)

        # The label endpoint is the date of the 21st-subsequent
        # INSTRUMENT bar, not the 21st calendar day after as_of. Map
        # the actual future date back to the global calendar via
        # ``shift(-horizon).get_indexer(...)``. Rows where the future
        # date is missing (instrument too short) get ``-1`` from
        # ``get_indexer`` and are filtered by ``valid_label_end_mask``.
        future_dates = group["date"].shift(-horizon_days)
        # pandas refuses NaT in DatetimeIndex via ``pd.DatetimeIndex(NaT)``
        # in some versions; build via the Series API which tolerates it
        # and yields ``NaT`` entries that ``get_indexer`` maps to ``-1``.
        label_end_indices = global_calendar.get_indexer(pd.DatetimeIndex(future_dates.to_numpy()))

        # Same trick to validate the 1d realized-return span: the next
        # instrument bar must be exactly the next global trading day.
        # If the instrument skipped a day, ``realized`` is silently a
        # 2-or-more-day return and would corrupt the daily MtM stream.
        next_dates = group["date"].shift(-1)
        next_bar_indices = global_calendar.get_indexer(pd.DatetimeIndex(next_dates.to_numpy()))
        valid_label_end_mask = label_end_indices >= 0
        valid_realized_span_mask = next_bar_indices == (as_of_indices + 1)

        feature_mat = group[feature_names].to_numpy(dtype=float)
        finite_mask = np.isfinite(feature_mat)
        has_any = finite_mask.any(axis=1)
        fwd_finite = np.isfinite(fwd)
        realized_finite = np.isfinite(realized)
        keep = (
            has_any
            & fwd_finite
            & realized_finite
            & (as_of_indices >= 0)
            & valid_label_end_mask
            & valid_realized_span_mask
        )
        if not keep.any():
            continue
        try:
            inst_uuid = uuid.UUID(str(inst))
        except ValueError:
            continue
        sector = sector_map.get(str(inst), "__unknown__")
        timestamps = group["date"].to_numpy()
        future_dates_np = future_dates.to_numpy()
        for idx in np.where(keep)[0]:
            features_row: dict[str, float] = {}
            row_finite = finite_mask[idx]
            for j, present in enumerate(row_finite):
                if present:
                    features_row[feature_names[j]] = float(feature_mat[idx, j])
            ts = pd.Timestamp(timestamps[idx]).tz_localize(UTC).to_pydatetime()
            label_end_ts = pd.Timestamp(future_dates_np[idx]).tz_localize(UTC).to_pydatetime()
            samples.append(
                SupervisedAlphaSample(
                    as_of=ts,
                    instrument_id=inst_uuid,
                    features=features_row,
                    forward_return=float(fwd[idx]),
                    metadata=(("sector", sector),),
                    realized_return_1d=float(realized[idx]),
                    as_of_index=int(as_of_indices[idx]),
                    label_end_index=int(label_end_indices[idx]),
                    label_end_as_of=label_end_ts,
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


def run_arm(
    arm_name: str,
    samples: list[SupervisedAlphaSample],
    feature_names: list[str],
    *,
    thresholds: AlphaEligibilityThresholds,
    portfolio_config: CampaignPortfolioConfig | None = None,
    fold_streak_risk_config: FoldStreakRiskConfig | None = None,
    model: AlphaModel | None = None,
    cost_model: TradingCostModel | None = None,
    weighting: WeightingScheme | None = None,
    selection: SelectionStrategy | None = None,
) -> tuple[WalkForwardEvidence, AlphaEligibilityThresholds, WalkForwardConfig]:
    # purge_days = HORIZON_DAYS is the audit-mandated minimum: a calendar
    # gap shorter than the label horizon lets training labels leak past
    # the purge into the test period. label_horizon_days makes the check
    # enforceable at config-construction time. The sample-level purge in
    # run_sample_walk_forward tightens this further on the trading-day
    # calendar.
    wf_config = WalkForwardConfig(
        train_window_days=252,
        test_window_days=21,
        step_days=21,
        purge_days=HORIZON_DAYS,
        embargo_days=0,
        min_folds=3,
        label_horizon_days=HORIZON_DAYS,
    )
    streak_tag = " +streak-dial" if fold_streak_risk_config else ""
    # ``model is None`` keeps the legacy linear ranker and its MODEL_VERSION so
    # arms A-H emit unchanged evidence; a supplied model stamps its own name.
    model_version = model.name if model is not None else MODEL_VERSION
    cost_tag = f", cost={cost_model.name}" if cost_model is not None else ""
    weight_tag = f", weighting={weighting.name}" if weighting is not None else ""
    select_tag = f", selection={selection.name}" if selection is not None else ""
    print(
        f"\n>>> Arm {arm_name}: {len(samples):,} samples, "
        f"{len(feature_names)} features, "
        f"portfolio={'long-only' if portfolio_config else 'signed-rank'}{streak_tag}, "
        f"model={model_version}{cost_tag}{weight_tag}{select_tag}, thresholds={thresholds.name}"
    )
    t0 = time.monotonic()
    evidence = run_sample_walk_forward(
        samples=samples,
        config=wf_config,
        model_version=model_version,
        feature_set_version=f"{FEATURE_SET_VERSION}--{arm_name}",
        thresholds=thresholds,
        slippage_bps_per_turnover=SLIPPAGE_BPS,
        feature_names=feature_names,
        weight_mode="ic_weighted",
        return_scale=1.0,
        portfolio_config=portfolio_config,
        fold_streak_risk_config=fold_streak_risk_config,
        model=model,
        cost_model=cost_model,
        weighting=weighting,
        selection=selection,
    )
    print(
        f"    {arm_name}: folds={len(evidence.folds)} "
        f"daily_obs={len(evidence.daily_returns):,} "
        f"({time.monotonic() - t0:.1f}s)"
    )
    return evidence, thresholds, wf_config


def _bucket_metrics(daily_returns: tuple[float, ...] | list[float]) -> dict[str, float]:
    """Derive the diagnostic non-overlapping 21d bucket metric block.

    The bucket series is *derived* from the canonical daily MtM stream by
    compounding simple returns inside each non-overlapping window —
    single source of truth. Reported alongside daily MtM but not gated.
    """
    bucket_returns = non_overlapping_bucket_returns(list(daily_returns), HORIZON_DAYS)
    return {
        "horizon_days": float(HORIZON_DAYS),
        "buckets": float(len(bucket_returns)),
        "total_return": compound_return(bucket_returns),
        "max_drawdown": max_drawdown(bucket_returns),
        "annualized_sharpe": bucket_sharpe(bucket_returns, HORIZON_DAYS),
    }


def _bars_snapshot_fingerprint(instrument_ids: list[str]) -> dict[str, object]:
    """Aggregate mtime+size fingerprint across the loaded bars parquet files.

    Hashes a deterministic listing of ``(path, size, mtime)`` triples
    sorted by path. Cheap (no file body read) but consistent across runs
    against the same on-disk snapshot. Renamed from ``..._hash`` to
    ``..._fingerprint`` because mtime+size cannot detect silent in-place
    mutation.
    """
    entries: list[str] = []
    for inst in sorted(instrument_ids):
        daily = BAR_ROOT / inst / "daily"
        if not daily.exists():
            continue
        for parquet_path in sorted(daily.glob("*.parquet")):
            try:
                stat = parquet_path.stat()
            except FileNotFoundError:
                continue
            entries.append(
                f"{parquet_path.relative_to(BAR_ROOT).as_posix()}|"
                f"{stat.st_size}|"
                f"{int(stat.st_mtime)}"
            )
    blob = "\n".join(entries).encode("utf-8")
    return {
        "algorithm": "sha256-of-path-size-mtime-listing",
        "is_content_hash": False,
        "files": len(entries),
        "fingerprint": hashlib.sha256(blob).hexdigest(),
    }


def _file_content_sha256(path: Path) -> str | None:
    """Full-body sha256 of ``path``, or ``None`` if the file is missing.

    Use for small, governance-relevant files (e.g. the universe
    definition) where a true content hash matters. For large
    parquet/bar snapshots, prefer ``_bars_snapshot_fingerprint`` —
    reading every body would be prohibitive.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def save_run_manifest(
    *,
    out_root: Path,
    run_id: uuid.UUID,
    started_at: datetime,
    finished_at: datetime,
    git_commit: str,
    cli_args_payload: dict[str, object],
    max_workers_used: int,
    requested_specs: list[ArmSpec],
    arm_results: list[tuple[ArmSpec, WalkForwardEvidence]],
    skipped_specs: list[tuple[ArmSpec, str]],
    universe_fingerprint: Mapping[str, object],
    bars_fingerprint: Mapping[str, object],
    regime_detector_meta: Mapping[str, object] | None = None,
) -> Path:
    """Write the run-level manifest alongside the per-arm evidence files.

    The manifest answers "what was this run?" without forcing a
    consumer to parse 7 per-arm evidence files. The COMPARISON table
    that the script prints to stdout is ephemeral; this manifest is
    the persistent index of the run.

    Output: ``<out_root>/run_manifest.json``. One file per run; later
    runs to the same ``out_root`` overwrite (the per-arm evidence
    files are also overwritten under the same naming).

    Schema is intentionally small — the manifest doesn't duplicate
    per-arm evidence; it points at it.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "run_manifest.json"

    def _arm_summary(spec: ArmSpec, evidence: WalkForwardEvidence) -> dict[str, object]:
        return {
            "cli_alias": spec.cli_alias,
            "canonical_name": spec.canonical_name,
            "category": spec.category,
            "production_candidate": spec.production_candidate,
            "panel_key": spec.panel_key,
            "evidence_file": f"arm_{spec.canonical_name}.json",
            "n_folds": len(evidence.folds),
            "eligibility_passed": bool(evidence.eligibility.get("passed", False)),
            # Headline metrics — duplicated from the per-arm evidence
            # so a manifest reader can answer "did anything pass?" /
            # "what's the spread of Sharpe?" without re-opening 7
            # files. The per-arm evidence remains the source of truth.
            "slippage_adjusted_sharpe": float(
                evidence.metrics.get("slippage_adjusted_sharpe", float("nan"))
            ),
            "max_drawdown": float(evidence.metrics.get("max_drawdown", float("nan"))),
            "total_return": float(evidence.metrics.get("total_return", float("nan"))),
            "fold_negative_ic_streak": float(
                evidence.metrics.get("fold_negative_ic_streak", float("nan"))
            ),
        }

    payload: dict[str, object] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "manifest_kind": "run",
        "run_id": str(run_id),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_clock_seconds": (finished_at - started_at).total_seconds(),
        "git_commit": git_commit,
        "cli_args": cli_args_payload,
        "max_workers_used": max_workers_used,
        "universe_fingerprint": universe_fingerprint,
        "bars_snapshot_fingerprint": bars_fingerprint,
        "requested_arms": [spec.cli_alias for spec in requested_specs],
        "completed_arms": [_arm_summary(spec, ev) for spec, ev in arm_results],
        "skipped_arms": [
            {
                "cli_alias": spec.cli_alias,
                "canonical_name": spec.canonical_name,
                "reason": reason,
            }
            for spec, reason in skipped_specs
        ],
    }
    if regime_detector_meta is not None:
        # Emitted at the manifest level whenever any arm in the run
        # consumed regime features — duplicates the per-arm block but
        # lets a manifest-only audit see the detector pinning without
        # opening any arm JSON. Deterministic from
        # DEFAULT_REGIME_THRESHOLDS so bit-identity across reruns
        # holds. Closes ADR-005 action item #10 (review finding #4).
        payload["regime_detector"] = dict(regime_detector_meta)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


def save_evidence(
    spec: ArmSpec,
    evidence: WalkForwardEvidence,
    out_root: Path,
    *,
    thresholds: AlphaEligibilityThresholds,
    wf_config: WalkForwardConfig,
    feature_set_versions: dict[str, str],
    universe_fingerprint: dict[str, object],
    bars_fingerprint: dict[str, object],
    git_commit: str,
    cli_args: dict[str, object],
    realized_mode_used: bool,
    regime_detector_meta: Mapping[str, object] | None = None,
) -> Path:
    """Write per-arm evidence JSON.

    Evidence field classification (for downstream bit-identity checks
    across reruns of the same inputs):

    * **Deterministic from inputs** — same inputs always produce same
      output: ``metrics``, ``metrics_daily_mtm``, ``metrics_bucket_21d``,
      ``folds``, ``daily_returns_count``, ``selected_weights``,
      ``feature_stability``, ``bootstrap_ic_ci``, ``portfolio_config``,
      ``portfolio_diagnostics``, ``drawdown_diagnostics``,
      ``eligibility``, ``eligibility_thresholds``, ``return_mode_daily``,
      ``return_mode_bucket``, ``fold_basis``, ``n_folds_actual``,
      ``label_horizon_days``, ``feature_set_versions``,
      ``walk_forward_config``, ``arm``, ``arm_cli_alias``,
      ``arm_category``, ``production_candidate``,
      ``evidence_schema_version``, ``slippage_bps_per_turnover``,
      ``realized_mode_used``, ``model_version``, ``feature_set_version``,
      ``regime_detector`` (regime arms only; pins detector thresholds
      + index-proxy strategy + breadth-source strategy so a future
      detector retune cannot silently invalidate this arm's
      evidence — see ADR-005 hardening / review finding #4).
    * **Varies per run** (must be excluded from strict-equality
      checks across reruns): ``run_id`` (uuid.uuid4 per run),
      ``saved_at_utc`` (wall-clock), ``cli_args.started_at_utc``,
      ``git_commit`` (varies across commits; deterministic at a
      pinned commit), ``bars_snapshot_fingerprint`` (varies if
      bars files were re-downloaded; deterministic at a pinned
      bars snapshot), ``universe_fingerprint.sha256`` (varies if
      universe file edited; deterministic otherwise).

    Bit-identity tests in ``test_backtest_latest_stack_parallel.py``
    exclude the varies-per-run set; the deterministic set is what
    ``--max-workers 1`` and ``--max-workers N`` must agree on.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"arm_{spec.canonical_name}.json"
    daily_returns_list = list(evidence.daily_returns)
    metrics_daily_mtm = dict(evidence.metrics)
    metrics_bucket_21d = _bucket_metrics(daily_returns_list)
    fold_basis_values = sorted(
        {str(fold.get("fold_basis", "calendar_days")) for fold in evidence.folds}
    )
    payload: dict[str, object] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "arm": spec.canonical_name,
        "arm_cli_alias": spec.cli_alias,
        "arm_category": spec.category,
        "production_candidate": spec.production_candidate,
        "run_id": str(evidence.run_id),
        "model_version": evidence.model_version,
        "feature_set_version": evidence.feature_set_version,
        "feature_set_versions": feature_set_versions,
        "label_horizon_days": HORIZON_DAYS,
        # Wording reflects the actual cadence: walk-forward folds are
        # generated on a *calendar-day* axis (test_window_days=21,
        # step_days=21 calendar days, roughly 15 trading days), and the
        # signed-rank evaluator rebalances once at the first as_of of
        # each fold. "21d rebalance" would mislead because it implies
        # trading-day fold boundaries — a deferred ADR-004 candidate.
        "return_mode_daily": (
            "realized_1d_simple_close_to_close_calendar_fold_rebalance_daily_mtm"
        ),
        "return_mode_bucket": "non_overlapping_21d_compounded_from_daily_mtm",
        # ``True`` when the evaluator actually ran in realized mode
        # (every sample carried ``realized_return_1d``). ``False`` would
        # mean a silent downgrade to legacy mode and the metrics CANNOT
        # be trusted as portfolio P&L — dashboards should filter these
        # out of any gated comparison.
        "realized_mode_used": realized_mode_used,
        "fold_basis": fold_basis_values,
        "n_folds_actual": len(evidence.folds),
        "git_commit": git_commit,
        "universe_fingerprint": universe_fingerprint,
        "bars_snapshot_fingerprint": bars_fingerprint,
        "eligibility_thresholds": asdict(thresholds),
        "walk_forward_config": asdict(wf_config),
        "cli_args": cli_args,
        "slippage_bps_per_turnover": evidence.slippage_bps_per_turnover,
        # ``metrics`` retains its existing semantics (eligibility-gated metrics
        # from the canonical daily MtM stream). ``metrics_daily_mtm`` aliases
        # it so dashboards can switch to the explicit name without losing the
        # legacy field, and ``metrics_bucket_21d`` is the diagnostic-only
        # derived bucket variant.
        "metrics": metrics_daily_mtm,
        "metrics_daily_mtm": metrics_daily_mtm,
        "metrics_bucket_21d": metrics_bucket_21d,
        "eligibility": dict(evidence.eligibility),
        "selected_weights": dict(evidence.selected_weights),
        "feature_stability": dict(evidence.feature_stability),
        "bootstrap_ic_ci": list(evidence.bootstrap_ic_ci),
        "folds": [dict(f) for f in evidence.folds],
        "portfolio_config": dict(evidence.portfolio_config),
        "portfolio_diagnostics": dict(evidence.portfolio_diagnostics),
        "drawdown_diagnostics": dict(evidence.drawdown_diagnostics),
        "daily_returns_count": len(daily_returns_list),
        "saved_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    if regime_detector_meta is not None:
        # Conditional emission — only regime arms carry the
        # detector-pinning block. Keeps the per-arm evidence schema
        # narrow for arms that don't touch regime features. Closes
        # ADR-005 action item #10 (review finding #4): pinning the
        # detector thresholds + index-proxy strategy in evidence so a
        # future detector retune cannot silently invalidate this
        # arm's metrics.
        payload["regime_detector"] = dict(regime_detector_meta)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


COMPARISON_METRICS = (
    "folds",
    "daily_observations",
    "oos_rolling_ic",
    "ic_60d",
    "slippage_adjusted_sharpe",
    "max_drawdown",
    "total_return",
    "turnover_avg",
    "fold_negative_ic_streak",
    "bootstrap_ic_p05",
    "bootstrap_ic_p95",
    "top_minus_bottom_decile_ic",
    # Fold-streak dial diagnostics (NaN for arms that don't use it).
    "fold_streak_scale_avg",
    "fold_streak_zero_fold_count",
)


# ---------------------------------------------------------------------------
# Parallel arm dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmJob:
    """Pickle-safe bundle of everything a worker needs to evaluate one arm.

    Lives at module scope (not nested in ``main()``) so
    ``ProcessPoolExecutor`` on Windows-spawn can re-import it cleanly.
    The heavy ``panel_df`` / ``close_panel`` payloads ride along by
    pickle; pandas serialises them in ~1-2s per arm against the
    universe-300 panel, which is dwarfed by the 4-7 min walk-forward
    cost — net we still get a 3-4x speedup at 4 workers.
    """

    spec: ArmSpec
    panel_df: pd.DataFrame
    feature_names: list[str]
    feature_set_versions: dict[str, str]
    close_panel: pd.DataFrame
    sector_map: dict[str, str]
    global_calendar: pd.DatetimeIndex
    pca_artifact_metadata: dict[str, object]
    universe_fingerprint: dict[str, object]
    bars_fingerprint: dict[str, object]
    git_commit: str
    out_root: Path
    cli_args_payload: dict[str, object]


@dataclass(frozen=True)
class _ArmJobResult:
    """Worker output. ``captured_stdout`` lets the main process print
    each arm's progress lines in a deterministic order even when the
    arms finish in a different order than they were submitted.

    Tagged-union shape: exactly one of ``evidence`` (success) and
    ``error`` (failure) is set. The ``__post_init__`` invariant
    enforces this so consumers don't need to defensively check both
    fields; ``unwrap_evidence()`` is the right way to access the
    evidence on the success path.
    """

    spec: ArmSpec
    evidence: WalkForwardEvidence | None = None
    captured_stdout: str = ""
    # Populated when the worker raised — ``main()`` surfaces these as
    # arm-skipped diagnostics rather than tearing down the whole run.
    error: str | None = None
    error_traceback: str = ""

    def __post_init__(self) -> None:
        if (self.evidence is None) == (self.error is None):
            raise ValueError(
                "_ArmJobResult must have exactly one of evidence/error set; "
                f"got evidence_is_none={self.evidence is None}, "
                f"error_is_none={self.error is None}"
            )

    def unwrap_evidence(self) -> WalkForwardEvidence:
        """Return the evidence; raise if this is an error result.

        Use after the caller has already branched on ``self.error``;
        the raise is a defensive guard, not a normal control path.
        """
        if self.evidence is None:
            raise RuntimeError(
                f"unwrap_evidence() called on error result for arm "
                f"{self.spec.cli_alias}: {self.error}"
            )
        return self.evidence


def _save_arm_evidence(
    spec: ArmSpec,
    evidence: WalkForwardEvidence,
    thresholds: AlphaEligibilityThresholds,
    wf_config: WalkForwardConfig,
    feature_set_versions: dict[str, str],
    *,
    realized_mode_used: bool,
    out_root: Path,
    pca_artifact_metadata: dict[str, object],
    universe_fingerprint: dict[str, object],
    bars_fingerprint: dict[str, object],
    git_commit: str,
    cli_args_payload: dict[str, object],
    regime_detector_meta: Mapping[str, object] | None = None,
) -> None:
    """Module-level evidence saver.

    Lifted from a closure in ``main()`` so the worker function can
    call it through a normal import path. Takes everything by
    parameter instead of by closure capture so it's safe to call
    from a child process. ``realized_mode_used`` is now an explicit
    parameter rather than derived from a ``samples`` list — keeps
    this function's surface focused on serialization and lets each
    caller make the realized-mode call against its own sample data.
    """
    if pca_artifact_metadata and "learned_pca_family" in feature_set_versions:
        feature_set_versions = {
            **feature_set_versions,
            "learned_pca_artifact": json.dumps(pca_artifact_metadata, sort_keys=True),
        }
    save_evidence(
        spec,
        evidence,
        out_root,
        thresholds=thresholds,
        wf_config=wf_config,
        feature_set_versions=feature_set_versions,
        universe_fingerprint=universe_fingerprint,
        bars_fingerprint=bars_fingerprint,
        git_commit=git_commit,
        cli_args=cli_args_payload,
        realized_mode_used=realized_mode_used,
        regime_detector_meta=regime_detector_meta,
    )


def _run_one_arm_job(job: _ArmJob) -> _ArmJobResult:
    """ProcessPoolExecutor worker: build samples + run walk-forward + save.

    Captures both stdout AND stderr so the main process can print
    each arm's output in deterministic order — without capture, prints
    from concurrently-running arms interleave and the log becomes
    unreadable. Stderr is included so warnings (``warnings.warn``
    output) and any logger-to-stderr writes also land in the per-arm
    buffer; both streams share one buffer so within-arm ordering is
    preserved. Exceptions are caught and returned as ``error`` on
    the result so a single arm's failure doesn't kill the pool.
    """
    import traceback  # noqa: PLC0415 — keep stdlib import local to worker

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            samples = build_supervised_samples(
                job.panel_df,
                job.close_panel,
                job.feature_names,
                job.sector_map,
                global_calendar=job.global_calendar,
            )
            portfolio_config = (
                job.spec.portfolio_config_factory() if job.spec.portfolio_config_factory else None
            )
            streak_config = (
                job.spec.fold_streak_risk_config_factory()
                if job.spec.fold_streak_risk_config_factory
                else None
            )
            # Built inside the worker (not pickled across the process boundary)
            # — the GBDT model holds a lazily-imported xgboost handle.
            model = job.spec.model_factory() if job.spec.model_factory else None
            # Cost model is cheap + picklable, but build it in the worker too so
            # all per-arm construction lives on one side of the boundary.
            cost_model = job.spec.cost_model_factory() if job.spec.cost_model_factory else None
            weighting = job.spec.weighting_factory() if job.spec.weighting_factory else None
            selection = job.spec.selection_factory() if job.spec.selection_factory else None
            # Look up the right eligibility threshold set for this
            # arm's category. ADR-004 records the per-category
            # governance contract; a missing category here raises
            # ``KeyError`` rather than silently defaulting.
            selected_thresholds = THRESHOLDS_BY_ARM_CATEGORY[job.spec.category]
            # ``run_arm`` returns the thresholds it received; we
            # already have ``selected_thresholds`` so we discard the
            # return-tuple slot (the explicit ``_`` makes the
            # round-trip relationship obvious to the reader).
            evidence, _returned_thresholds, wf_config = run_arm(
                job.spec.canonical_name,
                samples,
                job.feature_names,
                thresholds=selected_thresholds,
                portfolio_config=portfolio_config,
                fold_streak_risk_config=streak_config,
                model=model,
                cost_model=cost_model,
                weighting=weighting,
                selection=selection,
            )
            # Derive realized_mode_used from observed samples so the
            # evidence field reports a fact, not an assumption. The
            # latest-stack builder always populates realized_return_1d
            # today, so this is True in practice; the explicit check
            # future-proofs against a sample builder that drops the
            # field silently.
            realized_mode_used = all(s.realized_return_1d is not None for s in samples)
            # Pin detector thresholds + index-proxy strategy into the
            # arm's evidence iff the arm consumed regime features.
            # Computed in the worker so the value travels with the
            # per-arm payload; deterministic from DEFAULT_REGIME_THRESHOLDS
            # so bit-identity across reruns is preserved.
            regime_detector_meta = (
                regime_detector_metadata() if job.spec.panel_key == "pv_form_regime" else None
            )
            _save_arm_evidence(
                job.spec,
                evidence,
                selected_thresholds,
                wf_config,
                job.feature_set_versions,
                realized_mode_used=realized_mode_used,
                out_root=job.out_root,
                pca_artifact_metadata=job.pca_artifact_metadata,
                universe_fingerprint=job.universe_fingerprint,
                bars_fingerprint=job.bars_fingerprint,
                git_commit=job.git_commit,
                cli_args_payload=job.cli_args_payload,
                regime_detector_meta=regime_detector_meta,
            )
    except Exception as exc:
        return _ArmJobResult(
            spec=job.spec,
            captured_stdout=buffer.getvalue(),
            error=f"{type(exc).__name__}: {exc}",
            error_traceback=traceback.format_exc(),
        )
    return _ArmJobResult(
        spec=job.spec,
        evidence=evidence,
        captured_stdout=buffer.getvalue(),
    )


def _positive_int_argparse(value: str) -> int:
    """argparse ``type=`` for ``--max-workers``: integer >= 1.

    Rejects ``0`` and negative values at parse time with a usage
    error rather than silently falling through to the default in
    :func:`_resolve_max_workers`. ``1`` is the sequential mode.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"--max-workers must be >= 1 (use 1 for sequential); got {ivalue}"
        )
    return ivalue


def _resolve_max_workers(requested: int | None, n_arms: int) -> int:
    """Pick a sensible default worker count.

    Default: ``max(1, (os.cpu_count() or 2) // 2)``, clamped to the
    number of arms requested (no point spawning more workers than
    there are arms). Leaving half the cores free keeps the box
    responsive — the bottleneck is fold IC fitting, which already
    saturates a single thread; piling more workers on doesn't help
    once we're past CPU count.

    Caller is responsible for validating ``requested >= 1`` upstream
    (the CLI does this via :func:`_positive_int_argparse`); this
    function trusts the input and only handles the explicit-vs-
    default selection plus the n_arms clamp.
    """
    if n_arms <= 0:
        return 1
    if requested is not None:
        return min(int(requested), n_arms)
    cpu = os.cpu_count() or 2
    return max(1, min(cpu // 2, n_arms))


def print_comparison(
    arm_results: list[tuple[ArmSpec, WalkForwardEvidence]],
) -> None:
    print("\n" + "=" * 110)
    print("COMPARISON")
    print("=" * 110)
    # Use CLI alias for column headers — the canonical names are too long
    # for a console table.
    header = f"{'metric':<32}" + "".join(f"{spec.cli_alias:>17}" for spec, _ in arm_results)
    print(header)
    print("-" * len(header))
    for metric in COMPARISON_METRICS:
        row = f"{metric:<32}"
        for _, ev in arm_results:
            v = ev.metrics.get(metric, float("nan"))
            try:
                row += f"{float(v):>17.4f}"
            except (TypeError, ValueError):
                row += f"{str(v):>17}"
        print(row)
    print("-" * len(header))
    elig_row = f"{'eligibility.passed':<32}"
    for _, ev in arm_results:
        passed = ev.eligibility.get("passed", "?")
        elig_row += f"{str(passed):>17}"
    print(elig_row)
    cat_row = f"{'category':<32}"
    for spec, _ in arm_results:
        cat_row += f"{spec.category[:17]:>17}"
    print(cat_row)
    print("=" * 110)
    print("\nCanonical names:")
    for spec, _ in arm_results:
        print(f"  {spec.cli_alias} = {spec.canonical_name} ({spec.category})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Capture run-start metadata for the run-level manifest before any
    # work happens — even if arg parsing fails, the manifest path can
    # be reconstructed from the run_id.
    run_started_at = datetime.now(tz=UTC)
    run_id = uuid.uuid4()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-limit", type=int, default=None)
    parser.add_argument("--max-years", type=int, default=None)
    # Derive the default from ARM_SPECS so the next arm added to the
    # registry is in the default run automatically — prevents the
    # easy-to-miss bug where a new arm is declared but never runs
    # because the hardcoded default ``"A,B,C,D,E"`` wasn't updated.
    _default_arms = ",".join(spec.cli_alias for spec in ARM_SPECS)
    parser.add_argument(
        "--arms",
        default=_default_arms,
        help=(
            f"Comma-separated arm identifiers — either CLI aliases "
            f"({_default_arms}) or canonical names (research_ranker_pv, ...). "
            "Aliases and canonical names may be mixed."
        ),
    )
    parser.add_argument("--out-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--max-workers",
        type=_positive_int_argparse,
        default=None,
        help=(
            "Arm-level parallelism. Must be >= 1. ``1`` runs arms sequentially "
            "(same as the pre-parallel path; useful for debugging or "
            "pickle-safety checks). Default: ``max(1, cpu_count() // 2)`` "
            "clamped to the number of arms requested — leaves half the cores "
            "free for the OS while still giving ~3x wall-clock speedup on the "
            "7-arm run."
        ),
    )
    args = parser.parse_args(argv)

    requested_specs: list[ArmSpec] = []
    seen_aliases: set[str] = set()
    for raw in args.arms.split(","):
        key = raw.strip()
        if not key:
            continue
        spec = ARM_SPEC_BY_KEY.get(key)
        if spec is None:
            print(f"!! unknown arm identifier: {key!r} — skipping")
            continue
        if spec.cli_alias in seen_aliases:
            continue
        seen_aliases.add(spec.cli_alias)
        requested_specs.append(spec)
    print(
        "requested arms: "
        + ", ".join(f"{spec.cli_alias}={spec.canonical_name}" for spec in requested_specs)
    )

    # Reproducibility metadata: hash universe file content (cheap, deterministic)
    # and capture the current git commit. Bars fingerprint is computed after
    # bars load so the listing reflects only the instruments we actually used.
    git_commit = current_git_commit()
    universe_sha = _file_content_sha256(UNIVERSE_FILE) or "unavailable"
    universe_fingerprint = {
        "path": UNIVERSE_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": universe_sha,
    }
    # Snapshot the CLI arguments into the evidence so a future audit can
    # see exactly which subset of the universe/history the run covered.
    cli_args_payload: dict[str, object] = {
        "instrument_limit": args.instrument_limit,
        "max_years": args.max_years,
        "arms": args.arms,
        "out_root": str(args.out_root),
    }

    # 0. Universe
    print("[0] Loading universe ...")
    universe = load_universe()
    if args.instrument_limit is not None:
        universe = dict(list(universe.items())[: args.instrument_limit])
    sector_map = {k: str(v.get("sector", "__unknown__")) for k, v in universe.items()}
    print(f"    universe: {len(universe)} active instruments")

    # 1. Bars
    print("[1] Loading bars ...")
    t0 = time.monotonic()
    bars = load_bars(universe.keys(), max_years=args.max_years)
    n_inst = bars["instrument_id"].nunique()
    n_days = bars["date"].nunique()
    print(
        f"    bars: {len(bars):,} rows, {n_inst} instruments, {n_days} dates, "
        f"{bars['date'].min().date()} -> {bars['date'].max().date()} "
        f"({time.monotonic() - t0:.1f}s)"
    )
    close_panel = bars[["instrument_id", "date", "close"]].copy()
    bars_fingerprint = _bars_snapshot_fingerprint(list(universe.keys()))
    # Build the global trading-day calendar once so every arm's sample
    # builder uses identical as_of/label-end indices across arms.
    global_calendar = pd.DatetimeIndex(pd.Series(bars["date"].unique()).sort_values())

    # 2. Price-volume features
    print("[2] Computing price-volume features ...")
    t0 = time.monotonic()
    pv_df, pv_names = compute_pv_features(bars)
    print(f"    pv: {len(pv_df):,} rows × {len(pv_names)} features ({time.monotonic() - t0:.1f}s)")

    # 3. Formulaic alphas
    print("[3] Evaluating formulaic alpha library ...")
    t0 = time.monotonic()
    form_df, form_names = compute_formulaic_alphas(bars)
    print(
        f"    formulaic: {len(form_df):,} rows × {len(form_names)} alphas "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # Merge for arms B/C/D
    pv_form = pv_df.merge(form_df, on=["instrument_id", "date"], how="inner")
    source_names = pv_names + form_names
    print(f"    pv+formulaic merged: {len(pv_form):,} rows × {len(source_names)} features")

    # 4. Warmup PCA (only if any requested arm declares requires_pca=True).
    # Data-driven from ARM_SPECS so the next added PCA-requiring arm
    # doesn't re-trip the previous trap of a hardcoded ``{"C", "D"}``
    # set that silently skipped fitting for ``--arms E`` alone.
    full_panel = None
    full_names: list[str] = []
    pca_artifact_metadata: dict[str, object] = {}
    needs_pca = any(spec.requires_pca for spec in requested_specs)
    if needs_pca:
        pca_arm_aliases = sorted(spec.cli_alias for spec in requested_specs if spec.requires_pca)
        print(f"[4] Fitting warmup PCA artifact (needed by arms: {','.join(pca_arm_aliases)}) ...")
        # Warmup = first WARMUP_TRADING_DAYS of the panel
        unique_dates = pd.Series(pv_form["date"].unique()).sort_values().reset_index(drop=True)
        if len(unique_dates) <= WARMUP_TRADING_DAYS + 21:
            print(
                "    !! insufficient bars for warmup PCA — "
                f"skipping arms {','.join(pca_arm_aliases)}"
            )
        else:
            warmup_end = unique_dates.iloc[WARMUP_TRADING_DAYS - 1]
            t0 = time.monotonic()
            artifact = fit_warmup_pca_artifact(pv_form, source_names, warmup_end)
            print(
                f"    PCA: family={artifact.family_version}, "  # type: ignore[attr-defined]
                f"n_components={artifact.n_components}, "  # type: ignore[attr-defined]
                f"warmup_end={pd.Timestamp(warmup_end).date()}, "
                f"n_samples={artifact.fit_metadata['n_samples_fit']} "  # type: ignore[attr-defined]
                f"({time.monotonic() - t0:.1f}s)"
            )
            evr = artifact.explained_variance_ratio  # type: ignore[attr-defined]
            print(
                "    EVR per PC: " + ", ".join(f"{x:.3f}" for x in evr) + f"  (cum={sum(evr):.3f})"
            )
            pca_artifact_metadata = {
                "family_version": str(artifact.family_version),  # type: ignore[attr-defined]
                "n_components": int(artifact.n_components),  # type: ignore[attr-defined]
                "warmup_end": pd.Timestamp(warmup_end).date().isoformat(),
                "n_samples_fit": int(
                    artifact.fit_metadata["n_samples_fit"]  # type: ignore[attr-defined]
                ),
                "explained_variance_ratio": [float(x) for x in evr],
            }
            t0 = time.monotonic()
            learned_df, learned_names = compute_learned_features_panel(
                pv_form, source_names, artifact
            )
            print(
                f"    learned: {len(learned_df):,} rows × {len(learned_names)} features "
                f"({time.monotonic() - t0:.1f}s)"
            )
            full_panel = pv_form.merge(learned_df, on=["instrument_id", "date"], how="inner")
            full_names = source_names + learned_names

    # 5. Arms — data-driven dispatch. Each ArmSpec declares which prebuilt
    # panel it consumes and which portfolio config (if any) to apply.
    # Adding an arm is a single ARM_SPECS entry.
    arm_results: list[tuple[ArmSpec, WalkForwardEvidence]] = []
    skipped_specs: list[tuple[ArmSpec, str]] = []

    # Per-family feature-set versions for evidence reproducibility. Each
    # entry answers: "which feature-set version produced this column?".
    # Versions are imported from the real config modules so the evidence
    # cannot drift from the source of truth — a string mismatch here
    # would falsify the audit trail.
    fsv_pv: dict[str, str] = {"price_volume": PV_FEATURE_SET_VERSION}
    fsv_base: dict[str, str] = {**fsv_pv, "formulaic": FORMULAIC_FEATURE_SET_VERSION}
    # ``learned_pca_family`` is the feature-family version (the public
    # contract of column names + semantics); ``learned_pca_artifact_schema``
    # is the on-disk artifact schema version (the trainer/compute contract,
    # which bumped v1 → v2 in PR #61 when input standardisation was added).
    # Both are governance-relevant; emitting them separately keeps the
    # distinction legible in audit metadata.
    fsv_with_pca: dict[str, str] = {
        **fsv_base,
        "learned_pca_family": LEARNED_FEATURE_SET_VERSION,
        "learned_pca_artifact_schema": LEARNED_PCA_ARTIFACT_SCHEMA_VERSION,
    }
    fsv_with_regime: dict[str, str] = {
        **fsv_base,
        "regime": REGIME_FEATURE_SET_VERSION,
    }
    fsv_with_fund: dict[str, str] = {
        **fsv_base,
        "fundamentals": FUND_FEATURE_SET_VERSION,
    }

    # Panel registry: ``panel_key`` → (panel DataFrame, feature names,
    # feature_set_versions). ``full`` is only populated when PCA succeeded;
    # ``pv_form_regime`` only when at least one requested arm needs it
    # (regime computation costs ~30s on universe-300, so skip when no
    # arm consumes it).
    panel_registry: dict[str, tuple[pd.DataFrame, list[str], dict[str, str]]] = {
        "pv": (pv_df, pv_names, fsv_pv),
        "pv_form": (pv_form, source_names, fsv_base),
    }
    if full_panel is not None:
        panel_registry["full"] = (full_panel, full_names, fsv_with_pca)

    # 5b. Regime overlay panel (data-driven: only build when needed).
    # The regime family adds regime × base-feature interactions on
    # top of the pv+formulaic panel. See ADR-005 for the design.
    needs_regime = any(spec.panel_key == "pv_form_regime" for spec in requested_specs)
    if needs_regime:
        regime_arm_aliases = ",".join(
            spec.cli_alias for spec in requested_specs if spec.panel_key == "pv_form_regime"
        )
        print(f"[5b] Computing regime overlay features (arms: {regime_arm_aliases}) ...")
        t0 = time.monotonic()
        regime_ff = compute_regime_features(bars, pv_form)
        print(
            f"    regime: {len(regime_ff.frame):,} rows × {len(regime_ff.feature_names)} "
            f"features ({time.monotonic() - t0:.1f}s)"
        )
        # Merge regime features onto the pv+formulaic panel.
        regime_cols = list(regime_ff.feature_names)
        pv_form_regime = pv_form.merge(
            regime_ff.frame[["instrument_id", "date", *regime_cols]],
            on=["instrument_id", "date"],
            how="left",
        )
        # Training-feature set for arms consuming this panel: base
        # features + ONLY the interactions (not the indicators or
        # stats, which have IC=0 cross-sectionally).
        regime_training_names = [n for n in regime_cols if "__x__" in n]
        pv_form_regime_names = source_names + regime_training_names
        panel_registry["pv_form_regime"] = (
            pv_form_regime,
            pv_form_regime_names,
            fsv_with_regime,
        )

    # 5c. Fundamentals overlay panel (data-driven: only build when needed).
    # Adds the 9 Sharadar quality+value features (PIT-aligned) onto the
    # pv+formulaic panel — an alpha source orthogonal to momentum. Loading +
    # PIT-joining SF1 costs a few seconds, so skip when no arm consumes it.
    needs_fund = any(spec.panel_key == "pv_form_fund" for spec in requested_specs)
    if needs_fund:
        fund_arm_aliases = ",".join(
            spec.cli_alias for spec in requested_specs if spec.panel_key == "pv_form_fund"
        )
        print(f"[5c] Computing Sharadar quality+value fundamentals (arms: {fund_arm_aliases}) ...")
        t0 = time.monotonic()
        fund_df, fund_names = compute_fundamentals_panel(bars)
        fund_coverage = fund_df["instrument_id"].nunique() if not fund_df.empty else 0
        print(
            f"    fundamentals: {len(fund_df):,} rows × {len(fund_names)} features, "
            f"{fund_coverage} instruments covered ({time.monotonic() - t0:.1f}s)"
        )
        # Left-merge onto pv+formulaic: names without SF1 coverage keep their
        # pv+formulaic features and carry NaN fundamentals (neutral after rank).
        pv_form_fund = pv_form.merge(
            fund_df,
            on=["instrument_id", "date"],
            how="left",
        )
        pv_form_fund_names = source_names + fund_names
        panel_registry["pv_form_fund"] = (
            pv_form_fund,
            pv_form_fund_names,
            fsv_with_fund,
        )

    # Build the per-arm jobs from the panel registry. PCA-requiring arms
    # without a populated ``full`` panel are recorded as skipped here so
    # the parallel and sequential paths share the same skip semantics.
    arm_jobs: list[_ArmJob] = []
    for spec in requested_specs:
        if spec.requires_pca and "full" not in panel_registry:
            skipped_specs.append((spec, "PCA artifact unavailable"))
            print(f"    !! arm {spec.cli_alias} ({spec.canonical_name}) skipped — PCA unavailable")
            continue
        panel_df, feature_names_list, fsv = panel_registry[spec.panel_key]
        arm_jobs.append(
            _ArmJob(
                spec=spec,
                panel_df=panel_df,
                feature_names=list(feature_names_list),
                feature_set_versions=dict(fsv),
                close_panel=close_panel,
                sector_map=dict(sector_map),
                global_calendar=global_calendar,
                pca_artifact_metadata=dict(pca_artifact_metadata),
                universe_fingerprint=dict(universe_fingerprint),
                bars_fingerprint=dict(bars_fingerprint),
                git_commit=str(git_commit),
                out_root=Path(args.out_root),
                cli_args_payload=dict(cli_args_payload),
            )
        )

    # Dispatch: sequential when max_workers == 1 (or only one arm),
    # ProcessPoolExecutor otherwise. The sequential path is preserved
    # bit-for-bit so older saved transcripts diff cleanly against new
    # runs in --max-workers 1 mode. When no arms remain (all
    # PCA-requiring arms were skipped because PCA fitting failed), we
    # short-circuit — the ``if not arm_results`` branch below will exit.
    job_results: list[_ArmJobResult] = []
    max_workers = 1  # default if arm_jobs is empty (no dispatch needed)
    if arm_jobs:
        max_workers = _resolve_max_workers(args.max_workers, len(arm_jobs))
        print(
            f"\nDispatching {len(arm_jobs)} arms across "
            f"{max_workers} worker{'s' if max_workers != 1 else ''}"
            + (" (sequential)" if max_workers == 1 else " (parallel via ProcessPoolExecutor)")
        )
        if max_workers == 1:
            for job in arm_jobs:
                job_results.append(_run_one_arm_job(job))
        else:
            # ``submit`` + ``as_completed`` so each worker's stdout buffer
            # prints as soon as the arm finishes (still in completion order,
            # not submission order — the COMPARISON table re-sorts back to
            # the requested order below).
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_alias = {
                    executor.submit(_run_one_arm_job, job): job.spec.cli_alias for job in arm_jobs
                }
                for future in as_completed(future_to_alias):
                    job_results.append(future.result())

    # Stable presentation order: print each arm's captured stdout in
    # the order the user requested the arms (independent of completion
    # order), and append to ``arm_results`` likewise. Every result's
    # alias MUST be in ``spec_order`` — each job was built from a
    # requested spec — so a KeyError here is a real bug, not a
    # defensive fallback case.
    spec_order = {spec.cli_alias: idx for idx, spec in enumerate(requested_specs)}
    job_results.sort(key=lambda r: spec_order[r.spec.cli_alias])
    for result in job_results:
        # The buffered worker prints already end with their own
        # trailing newline from ``print``; emit them verbatim so the
        # console layout matches the sequential path.
        if result.captured_stdout:
            print(result.captured_stdout, end="" if result.captured_stdout.endswith("\n") else "\n")
        if result.error is not None:
            print(f"    !! arm {result.spec.cli_alias} failed: {result.error}")
            print(result.error_traceback)
            skipped_specs.append((result.spec, result.error))
            continue
        arm_results.append((result.spec, result.unwrap_evidence()))

    if not arm_results:
        print("no arms ran — nothing to compare")
        return 2

    # Run-level manifest. Written at run-end so each rerun overwrites
    # its predecessor for the same out_root. The manifest is the
    # index of the run; per-arm evidence files remain the source of
    # truth for each arm's metrics.
    finished_at = datetime.now(tz=UTC)
    # Pin detector metadata at the manifest level iff any arm in the
    # run consumed regime features. Closes ADR-005 action item #10
    # (review finding #4): a future detector retune cannot silently
    # invalidate this run's H/Shape-C evidence because the
    # thresholds + index-proxy strategy are now in the manifest.
    manifest_regime_meta: Mapping[str, object] | None = (
        regime_detector_metadata()
        if any(spec.panel_key == "pv_form_regime" for spec in requested_specs)
        else None
    )
    manifest_path = save_run_manifest(
        out_root=args.out_root,
        run_id=run_id,
        started_at=run_started_at,
        finished_at=finished_at,
        git_commit=git_commit,
        cli_args_payload=cli_args_payload,
        max_workers_used=max_workers,
        requested_specs=requested_specs,
        arm_results=arm_results,
        skipped_specs=skipped_specs,
        universe_fingerprint=universe_fingerprint,
        bars_fingerprint=bars_fingerprint,
        regime_detector_meta=manifest_regime_meta,
    )

    print_comparison(arm_results)
    print(f"\nEvidence JSON saved under: {args.out_root}")
    print(f"Run manifest: {manifest_path}")
    if skipped_specs:
        print("\n!! SKIPPED arms (return code 3):")
        for spec, reason in skipped_specs:
            print(f"    {spec.cli_alias} ({spec.canonical_name}): {reason}")
        # Non-zero exit so CI / operators notice the partial run instead
        # of treating it as a clean pass.
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
