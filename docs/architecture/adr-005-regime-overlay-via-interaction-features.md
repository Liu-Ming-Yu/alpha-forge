# ADR-005 — Regime overlay via interaction features

**Status:** Accepted (framework, hardened) — Shape-B' MVP shipped on branch `feat/regime-overlay-v1`; review findings #1–#5 closed on branch `feat/regime-overlay-v1-hardening` (v6.1). Empirical verdict from the original v6 rerun on universe-300: regime interactions did NOT move the binding `fold_negative_ic_streak` metric (4 → 4) and did NOT improve net Sharpe (1.0886 → 1.0511, −3.4%). They DID improve signal quality (bootstrap_ic_p05 +0.0028 → +0.0058) and Arm H still passes all 5 eligibility gates under `portfolio_candidate_v1`. **The v6.1 rerun supersedes the v6 numbers**: the parity fix (research now walks a shared `MarketRegimeDetector.step()` instead of threading `current_label` through `classify_regime`) changes the label sequence on regime-boundary dates; the oriented (anti-) interactions let the non-negative ranker express sign-flip priors. See the "Outcome" section for v6 numbers and "Hardening (post-review)" for v6.1 numbers + parity-fix rationale. Production lead remains G (long-only top-30 + streak dial, no regime); Shape C and follow-up ADR-006 stay deferred.
**Context:** v5 made G (long-only top-30 + streak dial, no PCA) eligible by passing all 5 gates under `portfolio_candidate_v1` — but with zero margin on streak (4 ≤ 4). A regime episode that adds one more negative-IC fold flips eligibility. The streak dial (ADR-003) cannot help: it throttles exposure, not predictions. To move the streak metric we need predictions that *adapt* to regime, not exposure that *reacts*. This ADR records the chosen shape of that adaptation.

## Context

The latest-stack alpha is an IC-weighted linear combination of price-volume + formulaic features. The IC fit is done once per walk-forward fold against the training window's forward returns. The same weights apply across the test window, regardless of any regime shift inside it.

A "regime" in this codebase is a 4-label classifier (`RISK_ON`, `RISK_OFF`, `TRANSITION`, `CRISIS`) derived from three market stats:

* `trend_z` — z-score of an index proxy vs its 200-day SMA
* `realized_vol` — annualised 21-day realised vol of the index
* `breadth` — fraction of instruments above their 50-day SMA

The detector is mature and live (signal service consumes it for execution sizing; portfolio service uses it for vol scaling). It has not been used as a research-feature input.

The empirical motivation: G's `fold_negative_ic_streak = 4` reflects a 4-fold (~84 trading days) stretch where the IC-weighted features were mis-aligned with realised forward returns. Most plausible cause: a regime shift inside the OOS window where the in-sample feature weights stopped working. The streak dial cuts exposure during that stretch (helps Sharpe, helps drawdown) but the underlying predictions remain wrong (streak metric unchanged at 7 in D, 4 in G).

For the streak metric to improve, the *predictions themselves* have to know about the regime.

## Decision

Ship a research feature family — `research/features/regime/` — that emits **regime × base-feature interactions** as additional columns to the panel. A new latest-stack arm (H) consumes the augmented panel and runs the same IC-weighted ranker; the ranker is then free to weight `feature_X × is_RISK_ON` differently from `feature_X × is_RISK_OFF`. This is the smallest change that gives the existing ranker regime-awareness without inverting the non-negative-weights design.

The MVP scope is deliberately conservative — a small selected subset of interactions, not the full cross-product — so the ranker doesn't overfit on a high-dimensional regime × feature space.

## Options considered

### Option A: Regime-gated exposure (multiplicative on weights)

Use the regime label to scale a long-only book's exposure (e.g., `gross_exposure × 0.5 if RISK_OFF`).

**Why deferred:**

- **Same shape as the streak dial.** Reactive overlay; cuts exposure once regime is observable but doesn't move the streak metric — exposure throttling cannot change cross-sectional IC.
- **Already partially implemented** via the streak dial and `portfolio_service/vol_sizing.py`. Stacking another reactive layer adds complexity without addressing the binding constraint.
- A regime-aware variant of the streak dial is a reasonable refinement, but only after the prediction-level adaptation lands.

### Option B: Raw regime indicators as features

Add 4 binary indicator columns (`is_RISK_ON`, `is_RISK_OFF`, `is_TRANSITION`, `is_CRISIS`) to the panel.

**Why deferred (rejected):**

- **IC = 0 by construction.** Regime is one label per date; every instrument on date `D` shares the same regime value. The IC-weighted ranker uses Spearman rank correlation *within each cross-section* (date). A constant column has uniform ranks → zero rank correlation → IC = 0 → dropped or zero-weighted by the ranker.
- This isn't a tuning failure; it's a mathematical inevitability of cross-sectional ranking.

### Option B' (chosen): Regime × base-feature interactions

For a curated set of `(base_feature, regime_label)` pairs, emit `base_feature × is_regime` as new feature columns. Each interaction varies cross-sectionally (because `base_feature` does) AND has stable IC *within its regime* (because the indicator concentrates the signal on regime-matching dates).

The ranker's IC fit then naturally rotates weights: `momentum × is_RISK_ON` and `momentum × is_RISK_OFF` get different in-sample ICs and accordingly different weights.

**Pros:**

- **Composable with the existing IC-weighted ranker.** No fitting-machinery changes; no per-regime model dispatch.
- **Compatible with `non_negative=True`** (the latest-stack default). Each interaction is its own column with its own non-negative IC; regime-overfit interactions get weight 0, surviving ones get positive weight.
- **Transparent in audit.** The `selected_weights` dict shows which interactions are being valued; an auditor reads "the ranker is weighting momentum × is_RISK_ON at 0.12 in this fold" directly.
- **Bounded feature growth.** Cross-product is `n_base × n_regimes = 36 × 4 = 144` if we did everything; the MVP curates a smaller set (e.g., 12 interactions covering the highest-conviction base features × 3 most distinct regimes) to keep degrees of freedom modest.

**Cons:**

- **Feature explosion if uncurated.** The full cross-product would be 144 columns added to the existing 36; that's a degrees-of-freedom multiplier that risks regime-overfit. Curation is required.
- **Interaction features inherit base-feature noise.** A noisy base feature × clean regime indicator = still noisy.
- **Discrete regime boundaries are sharp.** A continuous regime score (e.g., regime probability) would smooth transitions, but the existing `core/regime` infrastructure produces discrete labels with hysteresis-based stability smoothing. The MVP accepts the discrete shape; smoothing is a Shape D extension.

### Option C: Per-regime models

Fit `fit_correlation_weights` separately on the samples in each regime within the training window. At test time, route each test sample through the model for that sample's regime.

**Pros:**

- **Cleanest separation.** Each regime gets a model trained only on data from that regime; weights can be radically different across regimes without an interaction sign-flip needing to absorb that difference.
- **Theoretically the most expressive.** Anything Shape B' can do, Shape C can do (each interaction in B' corresponds to a sub-model in C); plus C can fit non-linear regime-specific dynamics.

**Why deferred:**

- **Data fragmentation.** A 252-day training window split 4 ways yields ~63 days per regime; some regimes may have <30 samples in some training windows. The IC fit becomes noisy or undefined.
- **Walk-forward refactor.** `run_sample_walk_forward` currently fits once per fold; supporting per-regime fits requires either (a) a per-fold-per-regime fitting loop with carry-state across regimes, or (b) routing each test sample at evaluation time through a regime-specific weight vector.
- **Right end-state, wrong starting point.** Once Shape B' demonstrates whether regime conditioning works at all, Shape C becomes a targeted refinement. Starting with C would conflate "does regime help?" with "is per-regime fitting necessary?"

**Promotion trigger:** ship Shape C as a follow-up ADR if Shape B' demonstrates that interaction features improve the streak metric AND the in-sample IC stability suggests data per regime is adequate (≥60 samples per regime per fold in typical windows).

### Option D: Regime probability smoothing

Instead of discrete `is_REGIME ∈ {0, 1}` indicators, use the regime classifier's confidence as a continuous `p_REGIME ∈ [0, 1]`. Each interaction becomes `base_feature × p_regime`.

**Why deferred:**

- **Existing detector emits discrete labels.** Producing calibrated regime probabilities is a non-trivial extension to the classifier (currently rule-based with a 3-day stability window for hard transitions).
- **Sharp boundaries are unlikely to be the dominant source of streak metric variance.** The 3-day stability window already smooths label transitions; a continuous probability would smooth at the margin of regime-shift dates, not in the middle of a 21-day fold.
- **Worth revisiting after Shape B'/C have run.** If the streak metric shows residual sensitivity to regime-boundary dates, calibrate probabilities then.

## Threshold values for the MVP

The regime detector reuses `DEFAULT_REGIME_THRESHOLDS` from `core/regime/thresholds.py`. The MVP does not retune the regime thresholds — those are calibrated against live use and have stability smoothing already.

The **curated interaction set** for the MVP (subject to revision based on evidence):

| base feature category | regimes used | rationale |
|---|---|---|
| Trend (e.g., `mom_12_1`, `ret_252d`) | `is_RISK_ON`, `is_RISK_OFF` | trend works in RISK_ON, reverses in RISK_OFF |
| Volatility (e.g., `vol_60d`, `realised_vol_21d`) | `is_RISK_ON`, `is_CRISIS` | low-vol anomaly varies sharply across regimes |
| Mean-reversion (e.g., `reversal_5d`) | `is_RISK_OFF`, `is_CRISIS` | reversion works strongest in stress regimes |

Approximate count: ~6 base features × 2-3 regimes = ~12-18 interactions. Plus the 4 raw regime indicators (kept for diagnostics; they'll get weight 0 but their presence in the panel lets a future ablation easily run with-vs-without).

## Implementation outline (this PR)

1. **`research/features/regime/`** — new family following the existing pattern (`config.py`, `features.py`, `__init__.py`).
   * `config.py`: `RegimeFeatureConfig` with the interaction-curation list, FEATURE_SET_VERSION = `"regime-v1"`.
   * `features.py`: `compute_regime_features(bars, base_features_panel)` returns a `FeatureFrame` of curated interactions + raw indicators.
   * Detector input: universe-mean price series as the index proxy (no SPY/QQQ dependency); per-instrument closes for breadth.
2. **Latest-stack panel extension**: `PanelKey` Literal gains `"pv_form_regime"`. The dispatch builds the regime panel by augmenting `pv_form` with regime interaction columns.
3. **Arm H**: `long_only_top30_pv_formulaic_streakdial_regime` — same construction as G plus the regime panel. Category `portfolio_candidate` (uses `portfolio_candidate_v1` thresholds).
4. **Tests**: regime label stability on synthetic stable bars, interaction shape (column count + alignment), zero-row defensive case, FamilyManifest registration.
5. **Universe-300 8-arm rerun**: A/B/C/D/E/F/G/H. Compare G vs H on streak metric + Sharpe.

## Consequences

### What becomes easier

- **Predictions can rotate by regime.** The IC-weighted ranker's weight vector becomes regime-aware via the interaction columns; no machinery change.
- **Audit story stays clean.** The `selected_weights` field in evidence shows interaction-level weights, so a reviewer can see exactly which regime is contributing.
- **Shape C (per-regime models) becomes the obvious next step** if Shape B' moves the streak metric. The interactions provide a low-cost first test of "does regime help at all?"; if yes, fragmenting the model per regime is a targeted refinement.

### What becomes harder

- **Feature curation matters.** The MVP picks a small interaction set by judgment, not by data. A full sweep over base × regime pairs would be a sub-ADR-level decision; the MVP is a starting point.
- **Regime detector becomes part of the research feature contract.** Today `core/regime` is consumed live. Adding research dependence means a future change to detector thresholds invalidates research evidence — pin the thresholds in evidence (`eligibility_thresholds`-style metadata).
- **Universe-mean as index proxy is one design choice.** If the universe shifts (e.g., universe-1000), the proxy mean is a different series. Evidence pins the universe; a future audit can reproduce given the universe.

### What we'll need to revisit

- **If Shape B' doesn't move the streak metric** — the regime classifier's signal isn't strong enough at this granularity, OR the interaction set is mis-curated, OR Shape C is required. Revisit by trying (in order) different curation, then Shape C.
- **If Shape B' moves Sharpe but not streak** — partial win; we have a defensible production_candidate with more margin on the streak gate. Worth shipping.
- **If Shape B' moves both Sharpe AND streak** — production_candidate gets stronger; G margin on streak widens. Promote to paper.
- **Continuous regime probability (Shape D)** — defer until Shape B'/C results.

## Outcome (v6 rerun, 2026-05-27)

Universe-300, 8 arms, 63 folds, same eligibility thresholds.

| Metric | G (no regime) | H (regime overlay) | Δ | Direction |
|---|---|---|---|---|
| `slippage_adjusted_sharpe` | 1.0886 | 1.0511 | −3.4% | worse |
| `max_drawdown` | −4.21% | −4.11% | +2.4% | slightly better |
| `total_return` | 0.1409 | 0.1334 | −5.3% | worse |
| `turnover_avg` | 0.48% | 0.57% | +18.8% | worse (cost drag) |
| `bootstrap_ic_p05` | +0.0028 | +0.0058 | +0.0030 | **better** |
| `bootstrap_ic_p95` | +0.0183 | +0.0222 | +0.0039 | better |
| `top_minus_bottom_decile_ic` | 0.0149 | 0.0141 | −0.0008 | tie |
| `fold_negative_ic_streak` | 4 | 4 | 0 | **unchanged** (binding gate) |
| `fold_streak_zero_fold_count` | 14 | 10 | −4 | dial firing less |
| Eligibility | pass | pass | — | both eligible |

What the ranker actually used (Arm H selected_weights, last fold; only non-zero interaction weights shown):

| Interaction | Weight | Direction |
|---|---|---|
| `reversal_5d × is_risk_off` | +0.1021 | classical mean-reversion-in-stress |
| `ret_252d × is_risk_on` | +0.0584 | 12m momentum bid in risk-on |
| `mom_6_1 × is_risk_off` | +0.0474 | momentum *survives* stress (unexpected) |
| `mom_12_1 × is_risk_on` | +0.0471 | 12-1 momentum in risk-on |
| `mom_6_1 × is_risk_on` | +0.0032 | mild |

5 of 12 curated interactions earned non-zero weight; 7 got zeroed by the `non_negative=True` ranker. The pattern is consistent with conventional academic priors (reversal in stress, momentum in risk-on), so the curation choices were directionally sane.

**Verdict:**

- The binding gate (`fold_negative_ic_streak`) did not move. The added cross-sectional variance from interactions is real (bootstrap_ic improved) but it doesn't fix the *same 4 folds* where the underlying IC inverts. The regime classifier either (a) labels those folds the same as the surrounding ones (no regime shift detected → no overlay help), or (b) labels them differently but the interaction set doesn't include the base features whose sign actually flipped.
- Net Sharpe and total return went slightly *down*. The +18.8% turnover from regime-conditional weight rotation outpaced the modest IC improvement.
- Both arms remain eligible under `portfolio_candidate_v1`. H is not worse for shipping; it is also not better than G.
- Framework is in place: `research/features/regime/`, the `pv_form_regime` panel, Arm H, FamilyManifest registration, detector-parity tests. Adding more regime experiments is now a configuration change.

**Failure-mode analysis:**

1. **Curation too narrow.** 12 interactions × 4 regimes possible = 16 untested combinations. The 4 folds where IC inverts may need a base feature we didn't pair with the matching regime.
2. **Sharp regime boundaries.** Discrete `is_REGIME ∈ {0, 1}` zeros out the interaction the instant the label transitions. Shape D (continuous probability) would smooth this — but only at transitions, not within-fold.
3. **Shape C may be required.** Per-regime models can rotate the *entire* weight vector across regimes (not just curated pairs), but data fragmentation in 252-day windows is a real concern (~63 days/regime).
4. **Streak gate may be uncalibrated.** Five arms (D/E/F/G/H) all park at 4; the gate may not have discriminative power below that. ADR-004 follow-up.

## Hardening (post-review, v6.1)

Review of the v6 PR surfaced five findings; this section records each one's resolution. The lesson across all five: discrete-indicator interactions on a non-negative ranker are a *narrow* hypothesis to test, and the original MVP implementation cut corners that made the test less rigorous than the ADR's parity claim implied.

### Finding #1 + #2 — research/live label parity was weaker than claimed

The v6 family called `classify_regime(stats, thresholds, current_label=previous_label)` directly and threaded `current_label = label` forward across the date loop. The live cycle uses `MarketRegimeDetector.detect(as_of)`, which has a *state machine*: a `deque` of candidate labels of length `stability_window` (default 3), a `_stable_regime` slot that only advances after 3 consecutive matching candidates, and a `disagree_haircut` applied to confidence when the stable and candidate labels diverge. Threading `current_label` is NOT equivalent to that machine — research labels switch faster than live labels on regime-boundary dates.

The v6 test `test_family_labels_match_classify_regime` only checked the family's label was *one of the valid four*, not that it equalled the live detector's label per date. It explicitly noted that a fresh single-date classification may disagree with the family's hysteresis-aware output.

**Resolution:**
- `MarketRegimeDetector` refactored: the async `detect()` body extracted into a private sync `_classify_with_state()`; new public sync `step(stats, as_of)` does `update(stats); _classify_with_state(as_of)` in one call. Live callers keep awaiting `detect()` (unchanged surface); research calls `step()` (same state machine).
- The research family's `_classify_per_date` now instantiates a single shared `MarketRegimeDetector` and calls `detector.step()` per date in order, guaranteed bit-for-bit parity with live.
- New test `test_family_label_sequence_matches_market_regime_detector_step` iterates every date in a synthetic 350-day panel, feeds the same canonical stats to a fresh `MarketRegimeDetector`, and asserts label equality per date. Zero tolerance, zero boundary slop. If parity ever breaks, this test fails before evidence is generated.

### Finding #3 — `non_negative=True` ranker cannot represent sign flips

ADR-005's curation included `mom_12_1 × is_risk_off`, `mom_6_1 × is_risk_off`, `ret_252d × is_risk_off` on the prior that "trend works in risk-on, reverses in risk-off." The IC-weighted ranker runs with `non_negative=True` and clips negative in-sample IC to weight 0. If the prior is right and trend genuinely *reverses* in risk-off, those positive-orientation columns have negative IC inside the regime and are silently dropped — the regime axis can only condition *strength*, not *direction*.

**Resolution:**
- New `RegimeInteractionSpec(base_feature, regime_label, negate: bool)` dataclass in `config.py`. `negate=False` emits `f"{base}__x__{regime}"` carrying `base × indicator`; `negate=True` emits `f"anti_{base}__x__{regime}"` carrying `-base × indicator`. The naming prefix lets the auditor distinguish orientations in `selected_weights` without parsing the spec.
- `features.py` updated to consume the new spec and emit the sign multiplier (`±1.0 * base * indicator`).
- `DEFAULT_INTERACTIONS` re-curated: positive trend × risk_off replaced with anti-variants (`anti_mom_12_1__x__risk_off`, `anti_mom_6_1__x__risk_off`, `anti_ret_252d__x__risk_off`). Other priors (reversal in stress, vol-anomaly) stay positive-orientation. Total interaction count unchanged at 12, so the ranker's degrees of freedom are bit-identical.
- New tests: `test_negated_interaction_emits_anti_prefix`, `test_negated_interaction_zero_outside_regime`, `test_negated_interaction_equals_minus_base_inside_regime`, `test_curation_includes_both_orientations` (curation invariant — at least one negated interaction must be present, else this whole hardening is undone in a future bad edit).

### Finding #4 — evidence did not pin regime-detector thresholds

ADR-005 already named this as a follow-up (original action item #10). v6 evidence had no `regime_detector` block; a future retune of `DEFAULT_REGIME_THRESHOLDS` would silently invalidate Arm H's metrics.

**Resolution:**
- New `regime_detector_metadata()` helper in `research/features/regime/features.py` returns a JSON-safe dict with: `feature_set_version`, `detector_version` (the SHA-8 hash of thresholds from `core.regime.detector_version`), `thresholds` (all 9 `RegimeThresholds` fields exhaustively pinned by name), `index_proxy` (`"universe_mean_close"`), `breadth_source` (`"per_instrument_close_vs_50d_sma"`).
- `save_evidence` emits the block into per-arm JSON when the arm consumed regime features (panel_key=`"pv_form_regime"`).
- `save_run_manifest` emits the block at the manifest level when any arm in the run consumed regime features.
- Evidence-field classification docstring updated: `regime_detector` is in the "deterministic from inputs" bucket — bit-identical across reruns at a pinned commit.
- New tests: `test_metadata_keys_match_audit_contract`, `test_metadata_thresholds_exhaustively_pin_RegimeThresholds`, `test_metadata_detector_version_matches_core_helper`.

### Finding #5 — H's turnover drag is real, not cosmetic

v6 turnover rose +18.8% (G 0.48% → H 0.57%) while Sharpe fell −3.4%. The mechanism is mechanical: discrete `is_REGIME ∈ {0, 1}` indicators flip the entire interaction-column state at a regime boundary, scores and ranks shift, the rebalancer churns. This is independent of whether the regime *prediction* is right — it's purely a coupling between discrete regime boundaries and a continuous ranker.

**Resolution:** *Documented as a known failure mode of the Shape-B' MVP*; no code change in v6.1 because mitigating it would require either:

- **Continuous regime probabilities** (Shape D from the Options Considered) — requires the live detector to emit calibrated probabilities, currently rule-based with hysteresis. Non-trivial extension; defer.
- **Centered interactions** — emit `base × (indicator - regime_frequency)` so the column is zero-mean over the full history rather than zero on out-of-regime dates. Reduces the discontinuity at regime boundaries. Worth trying as a one-line change in a follow-up if H is ever revisited.
- **Turnover-aware feature selection** — penalise interactions that contribute disproportionately to rebalance churn. Couples feature-side decisions to the trading layer; bigger refactor.

Recorded in this ADR rather than in code so the next person who looks at Arm H knows why H's net Sharpe slipped without re-running an ablation.

## Action Items

1. [x] Write `research/features/regime/` family — `config.py`, `features.py`, `__init__.py` with `FamilyManifest`.
2. [x] Implement `compute_regime_features()` — calls canonical `compute_market_stats` per-date for live-vs-research parity. Universe-mean index proxy. Output: 4 indicators + 3 stats + 12 interactions.
3. [x] Curate the interaction set (12 columns: trend × {risk_on, risk_off}, mean-reversion × {risk_off, crisis}, vol × {risk_on, crisis}).
4. [x] Extend `PanelKey` Literal with `"pv_form_regime"`; add panel construction (lazy-built only when an Arm H-class arm is requested).
5. [x] Add `ArmSpec` for Arm H to `ARM_SPECS`.
6. [x] Unit tests: 15 tests covering manifest, indicators, cross-sectional invariance, interaction shape, missing-base fallback, empty-bars, stability, and detector-parity (stats + labels bit-for-bit match `core/regime`).
7. [x] Universe-300 8-arm rerun. G vs H captured above.
8. [x] Memory `[Backtest Latest Stack]` v6 section landed.
9. [ ] **Shape C / ADR-006 deferred** — the MVP delivered enough signal-quality lift to confirm the regime axis is real, but not enough Sharpe/streak lift to justify the per-regime-model refactor right now. Better next moves in priority order: (a) walk-forward-tune the streak dial defaults (ADR-004 #12, pre-paper hardening), (b) ship the latest-stack → registry adapter so G can enter paper (ADR-004 #14), (c) only then revisit Shape C if a streak-margin gap reappears.
10. [x] **Regime-detector thresholds pinned in evidence + manifest** — closed in v6.1 via `regime_detector_metadata()`; emitted into per-arm evidence (regime arms only) and into the run manifest (whenever any regime arm in the run).
11. [x] **Stateful detector parity (v6.1, review #1+#2)** — research family now walks a shared `MarketRegimeDetector.step()` per date; sequence-level parity asserted in `test_family_label_sequence_matches_market_regime_detector_step`.
12. [x] **Oriented (anti-) interactions (v6.1, review #3)** — `RegimeInteractionSpec.negate=True` emits `-base × indicator` so the `non_negative=True` ranker can express sign-flip priors; curation invariant pinned in `test_curation_includes_both_orientations`.
13. [ ] **Continuous regime probabilities (Shape D) or centered interactions** — defer until a regime workstream is revisited; documented as the future-work direction for the turnover-drag failure mode (review #5).

## Related

- [ADR-003](adr-003-return-accounting-separation.md) — Return-accounting separation; named "regime overlay" as the next big lift after the streak dial.
- [ADR-004](adr-004-per-category-eligibility-thresholds.md) — Per-category eligibility thresholds; made G eligible but with zero margin on streak. This ADR addresses the margin.
- [ADR-002](adr-002-learned-family-representation-choice.md) — Learned-family representation. Explicitly noted: "ship a regime-detector family first, then condition the learned transform on it in v2." This ADR ships the regime-detector family.
- `core/regime/` — existing 4-label classifier with stability smoothing + hysteresis. Reused as-is.
- Memory: [Backtest Latest Stack](../../memory/project_backtest_latest_stack.md) — v5/v5.1 baseline against which Arm H is compared.
