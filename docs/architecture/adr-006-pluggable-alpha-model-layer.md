# ADR-006 — Pluggable alpha-model layer (and the GBDT ranker experiment)

**Status:** Accepted (framework + rank arm). The model seam (`campaigns/models/`) is shipped and is the durable win. Two non-linear learners followed: **Arm I** (GBDT, MSE objective) is NOT eligible — it fails `ic_60d` (0.0278 < 0.03) because MSE-on-levels is the wrong loss for a ranker. **Arm J** (GBDT, `rank:pairwise` + per-date query groups) **IS eligible** — switching to a ranking loss recovered IC quality (ic_60d 0.0278 → 0.0641, decile spread 0.0008 → 0.0103) and J passes all 5 gates with the **highest total return of any arm (17.1%)**. **But all four model/overlay interventions — streak dial (ADR-003), regime overlay (ADR-005), GBDT-MSE, GBDT-rank — leave `fold_negative_ic_streak` at exactly 4**, which is now strong evidence the streak is a data property, not a model limitation. Production lead remains **G** (highest Sharpe 1.0886, highest IC 0.2561, tightest DD −4.21%); J is a validated eligible challenger, strongest on raw return.

**Context:** the qlib model-adoption review identified swapping the linear IC-weighted ranker for a non-linear learner as the highest-leverage path to move the binding `fold_negative_ic_streak` gate — the constraint that the exposure dial (ADR-003) and the regime overlay (ADR-005) both failed to move. The driver inlined a single ranker, so testing that hypothesis first required a model seam.

## Context

`run_sample_walk_forward` refits a model on each (purged) walk-forward fold and scores the test cross-section. Historically the only "model" was the linear IC-weighted ranker — per-feature Spearman IC, normalized to weights, scored as a weighted sum — inlined in the driver via `fit_correlation_weights` / `score_features`.

The linear ranker's structural weakness is that it cannot represent feature **interactions**: momentum that only pays in a low-volatility regime, reversal that only works under stress. The regime overlay (ADR-005) tried to inject a curated handful of interactions as explicit columns; it improved signal-quality bounds but did not move the streak gate. A tree ensemble learns interactions natively — the natural next experiment, and the one qlib's model zoo is built around (LightGBM/XGBoost, then sequence and graph models).

To run that experiment without destabilizing the audited machinery (sample-level purge, eligibility gate, portfolio constructor, streak dial), the fit/score boundary had to become pluggable.

## Decision

Factor the per-fold fit/score boundary into two small protocols in `campaigns/models/`:

* `AlphaModel.fit(train, feature_names) -> FittedAlphaModel` — a per-fold **factory**.
* `FittedAlphaModel.score(samples) -> list[float]` + `feature_weights() -> Mapping[str, float]` — an **immutable, fold-local** fitted object.

Each fold gets a fresh fit, so a fitted object never carries state across folds — the point-in-time contract is impossible to violate by accident. `run_sample_walk_forward` gains `model: AlphaModel | None = None`; `None` builds the behavior-preserving `LinearICRanker` (default arms A–H stay bit-identical). Everything downstream is model-agnostic and untouched.

Two concrete models ship:

* **`LinearICRanker`** — wraps the existing IC functions; the default. Its `name` matches the legacy `MODEL_VERSION` so arm evidence is unchanged.
* **`GradientBoostedRanker`** — XGBoost, `device="auto"` (CUDA probe, CPU fallback). Normalized gain importances serve as the `feature_weights` reporting proxy so the `selected_weights` evidence field and cross-fold `feature_stability` stay well-formed for tree models.

A new arm **I = G's exact construction (no-PCA pv+formulaic panel, long-only top-30, streak dial) with the linear ranker swapped for the GBDT** gives a clean A/B against the production lead.

## Options considered

### XGBoost vs LightGBM
Chose **XGBoost**: already a project dependency (the `ml` extra) with first-class CUDA support on Windows out of the box. GPU LightGBM needs a custom OpenCL/CUDA build. The architectural point — a boosted-tree ranker behind the `AlphaModel` protocol — is identical; a LightGBM model would be a drop-in sibling of `gbdt.py`.

### Two-protocol (factory → fitted) vs single stateful model
Chose factory → fitted so the fitted object is immutable and fold-local; no risk of state bleeding across folds. Cost: one extra class per model. Worth it for the PIT guarantee.

### Feature importances as the `selected_weights` proxy
A tree model has no linear coefficients. Using normalized **gain** importances keeps `feature_stability` (stability of importance ranking across folds) meaningful and the evidence schema stable. Trade-off: `selected_weights` semantics shift from "linear coefficient" to "importance" for GBDT arms — disambiguated by `model_version` in the evidence.

### MSE objective first
`reg:squarederror` on the forward-return level is the direct non-linear analog of IC-weighting and the simplest starting point. Ranking objectives (`rank:pairwise` / `rank:ndcg`) need per-date query groups and were deferred — see Consequences.

### GPU determinism
GPU `hist` is approximately (not bit-) deterministic. Acceptable: Arm I is a new arm, not bit-compared against frozen history. The `name` (model_version) is hardware-independent so reruns on different machines remain comparable.

## Outcome (universe-300, 63 folds, 907 OOS days, commit `9d34da67`)

| metric | G (linear) | I (GBDT-MSE) |
|---|---:|---:|
| slippage_adjusted_sharpe | **1.0886** | 1.0387 |
| max_drawdown | −0.0421 | −0.0471 |
| total_return | 0.1409 | **0.1601** |
| turnover_avg | 0.0048 | 0.0086 |
| **fold_negative_ic_streak** | **4** | **4** |
| oos_rolling_ic | 0.2561 | 0.1192 |
| ic_60d | 0.0912 | **0.0278** (gate 0.03) |
| bootstrap_ic_p05 | 0.0028 | 0.0002 |
| top_minus_bottom_decile_ic | 0.0149 | 0.0008 |
| **eligibility.passed** | **True** | **False** (fails `ic_60d`) |

A–H reproduce the v6.1 numbers bit-identically; the model seam did not perturb the default path.

### Three findings

1. **The binding streak gate did not move (4 → 4).** Three independent interventions — the exposure dial (ADR-003), regime interactions (ADR-005), and now a non-linear GBDT — all failed to move `fold_negative_ic_streak`. This is now strong evidence the 4-fold negative-IC streak is a **data property** (a real regime episode inside the OOS window), not a model-capacity limitation. The streak is unlikely to yield to *any* model trained on this feature set; it more plausibly needs a different universe, a longer/short construction, or an explicit regime-conditioned label.

2. **MSE-on-levels is the wrong objective for a ranker.** The GBDT delivers *higher* total return (16.0% vs 14.1%) and comparable Sharpe (1.04), but markedly worse rank quality: `oos_rolling_ic` halved (0.119 vs 0.256), `ic_60d` cut to a third (0.028 vs 0.091), decile spread collapsed (0.0008 vs 0.0149). It learned to pick a few winners (return) but not to *rank the cross-section* (IC), and fails eligibility on `ic_60d` by a hair. The qlib lesson lands precisely here: its strongest models gate on **IC / rank loss**, not MSE.

3. **The infrastructure is the lasting win.** The model seam unlocks every future model experiment — rank-loss GBDT, GRU/ALSTM, TFT — behind one protocol, with zero churn to the leakage, eligibility, portfolio, or dial machinery. The negative empirical result for the MSE arm does not diminish the architectural value.

## Consequences

* **Production lead remains G.** Arm I is a diagnostic (not a candidate); Arm J is an eligible challenger but does not beat G on risk-adjusted quality. Neither displaces G yet.
* The default arm set now includes I and J (both require the `ml` extra / xgboost). When xgboost is absent the worker reports the arm as errored rather than tearing down the run.
* Arms A–H are unchanged and bit-identical; the linear ranker is now `LinearICRanker` but produces the same evidence.

### Outcome — Arm J: rank-loss GBDT (shipped)

`GradientBoostedRanker` gained `objective="rank"` (`rank:pairwise`). The driver passes `train` date-sorted, but the model also sorts by `as_of` defensively and derives per-date query groups via `DMatrix.set_group`. Arm J = G + GBDT(rank), same panel/construction/dial as I, only the objective differs.

| metric | G (linear) | I (GBDT-MSE) | J (GBDT-rank) |
|---|---:|---:|---:|
| slippage_adjusted_sharpe | **1.0886** | 1.0387 | 1.0524 |
| max_drawdown | **−0.0421** | −0.0471 | −0.0518 |
| total_return | 0.1409 | 0.1601 | **0.1707** |
| **fold_negative_ic_streak** | **4** | **4** | **4** |
| oos_rolling_ic | **0.2561** | 0.1192 | 0.1979 |
| ic_60d | 0.0912 | 0.0278 | 0.0641 |
| top_minus_bottom_decile_ic | 0.0149 | 0.0008 | 0.0103 |
| **eligibility.passed** | **True** | False | **True** |

Findings:

1. **Rank loss fixes the GBDT, exactly as the Arm I diagnosis predicted.** Switching MSE → `rank:pairwise` more than doubled `ic_60d` (0.0278 → 0.0641, above the 0.03 gate), recovered the decile spread 13× (0.0008 → 0.0103), and lifted `oos_rolling_ic` (0.119 → 0.198). J passes all five gates. This is the qlib lesson confirmed empirically: **gate the model on ranking/IC loss, not MSE.**
2. **J is a validated eligible challenger** — the third eligible arm (with G and H) and the **highest total return of any arm (17.1%)**. G still leads on risk-adjusted quality (Sharpe 1.089 vs 1.052, IC 0.256 vs 0.198, DD −4.2% vs −5.2%), so it remains the production lead, but J is a real alternative when raw return is weighted more heavily.
3. **The streak STILL did not move (4).** Four independent interventions — exposure dial, regime interactions, GBDT-MSE, GBDT-rank — all leave `fold_negative_ic_streak` at exactly 4. This is now compelling evidence the 4-fold streak is a **data property** (a specific OOS regime episode common to every model on this feature set), not a model-capacity limitation. Moving it likely requires a different universe, label, or long/short construction — not a better in-family model. This reframes the eligibility-margin risk for G/H/J: a one-fold worsening of that episode would flip all three at once, so the streak threshold itself (ADR-004) is the lever to revisit, not the model.

Sequence models (GRU/ALSTM via torch+GPU — where GPU becomes genuinely necessary) remain the larger follow-up under the same protocol, but finding #3 lowers their expected value for the *streak* specifically.

**Related:** ADR-003 (return accounting + streak dial), ADR-005 (regime overlay — the other intervention that did not move the streak gate), the qlib model-adoption review.
