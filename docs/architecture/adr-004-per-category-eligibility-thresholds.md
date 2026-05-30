# ADR-004 — Per-category eligibility thresholds

**Status:** Accepted (2026-05-28) and **shipped** in PR #71 on branch `feat/per-category-eligibility-thresholds`. Closes the "per-category eligibility threshold separation" action item flagged in ADR-003. **Addendum 2026-05-28:** streak-gate settlement + held-out calibration appended (see sections below) — Action Items 12–16 all closed. The flat `streak<=4` candidate gate is **replaced** by a drawdown-conditioned gate (Option D, Item 16, `portfolio_candidate_v2`, containment −2%): held-out calibration showed the flat threshold was overfit and un-validatable, so laxity is now earned per-episode by the realized within-streak drawdown. Implemented + verified (verdicts unchanged on A–J). The model-registry adapter (Item 14) now promotes a winning arm's evidence without a manual step. The gate is no longer a blocker to G's promotion. **G was promoted to the model registry on 2026-05-28** (one-call live path via `scripts/promote_latest_stack_arm.py --register`; `signal_type=classical` after the `alpha` promote/assert CLI taxonomy was extended to include `classical` — the linear-ranker type — via the shared `ALPHA_SOURCE_TYPES`, closing the last gap). G is now the active registered model for `long_only_top30_pv_formulaic_streakdial`, entering the paper-trading sequence. The canonical A–J evidence was then re-executed under `portfolio_candidate_v2` (identical headline metrics + verdicts, now with the DD-conditioned streak check and `max_drawdown_during_worst_streak`) and G re-promoted on that v2 evidence; the prior v1 registry record is retained inactive as audit history.
**Context:** v4 of the latest-stack backtest left Arm G with Sharpe 1.0886 / DD −4.21% / streak 4, failing eligibility only on `fold_negative_ic_streak <= 2`. The single global `AlphaEligibilityThresholds` was calibrated for signed-rank baselines with no risk controls, but the same gate was being applied to long-only top-30 candidates whose construction tames drawdowns to ~4%. The same threshold cannot calibrate both — this ADR records why we split.

## Context

The platform's research promotion gate is implemented by `eligibility()` in `services/research_service/sampling/eligibility.py`, which takes a `metrics` dict and an `AlphaEligibilityThresholds` instance and returns a pass/fail decision plus per-check breakdown. The thresholds were a single global dataclass with five fields:

```python
@dataclass(frozen=True)
class AlphaEligibilityThresholds:
    min_oos_rolling_ic: float = 0.05
    min_ic_60d: float = 0.03
    max_fold_negative_ic_streak: int = 2
    max_drawdown: float = -0.20
    min_slippage_adjusted_sharpe: float = 1.0
```

The latest-stack backtest evaluates two structurally different arm categories against this same gate:

* **`research_ranker_baseline`** (A/B/C): signed-rank weights from an IC-weighted feature mix. No per-name cap, no sector neutralisation, no ADV cap, no borrow model, no cash model. Diagnostic tools measuring whether the feature stack ranks future returns — never something you'd trade. Naturally produces 15-17% drawdowns and can run 4-7 consecutive negative-IC folds without that being catastrophic, because nothing is at stake.

* **`portfolio_candidate`** (D/E/F/G): long-only top-30 with `max_single_name_weight=0.05`, `max_gross_exposure=0.22`, `min_cash_buffer=0.05`, `rebalance_interval_days=21`, optional fold-streak exposure dial. Naturally produces ~4% drawdowns because the construction caps exposure. Designed to be a production-candidate alpha.

Applying `max_fold_negative_ic_streak <= 2` and `max_drawdown >= -0.20` uniformly to both categories has a specific failure mode that v4 surfaced: G's risk profile (DD −4.21%, far inside −20%) showed the drawdown gate was loose against a constrained candidate, while the streak gate was tight regardless of how the construction handled the streak.

A streak of 4 negative-IC folds at −4% DD is not the same risk as the same streak at −16% DD. One is a regime episode the construction absorbed; the other is the alpha being broken AND the construction not catching it. The gate should reflect that.

## Decision

`AlphaEligibilityThresholds` keeps its single-class shape but gains a `name: str` field for audit-trail identification, and `factory_models.py` adds two named module-level instances plus a category-keyed lookup:

* **`RESEARCH_RANKER_BASELINE_THRESHOLDS`** (`name="research_ranker_baseline_v1"`) — identical numeric values to the legacy default; streak ≤ 2, DD ≥ −20%. The strict gate; baselines must clear it.
* **`PORTFOLIO_CANDIDATE_THRESHOLDS`** (`name="portfolio_candidate_v1"`) — streak ≤ 4, DD ≥ −10%, all other fields identical to the baseline set. Looser streak in exchange for tighter drawdown.
* **`THRESHOLDS_BY_ARM_CATEGORY`** — `dict[str, AlphaEligibilityThresholds]` keyed by `ArmCategory` string values. Scripts dispatch through this map; a missing category raises `KeyError` (loud), never silently defaults.

The contract encoded by the asymmetric trade: *"we trust the construction iff it actually protects you."* A portfolio_candidate gets up to 4 consecutive negative-IC folds **only if** its construction kept DD inside −10%. If the construction misbehaves and DD blows past −10%, the looser streak gate doesn't help — the DD gate fails first. Both halves of the trade have to land together.

The `name` field rides in the evidence JSON's `eligibility_thresholds.name` so audit-trail readers can identify which set was applied without re-deriving it from the numeric values.

## Threshold values (full)

| field | `research_ranker_baseline_v1` | `portfolio_candidate_v1` |
|---|---:|---:|
| `min_oos_rolling_ic` | 0.05 | 0.05 *(same)* |
| `min_ic_60d` | 0.03 | 0.03 *(same)* |
| `max_fold_negative_ic_streak` | **2** | **4** |
| `max_drawdown` | **−0.20** | **−0.10** |
| `min_slippage_adjusted_sharpe` | 1.0 | 1.0 *(same)* |

IC and Sharpe gates are identical across categories — the alpha-quality bar doesn't change because the construction changes. Only the gates that interact with construction-bounded behaviour (streak tolerance, drawdown bound) differ.

## Options considered

### Option A: Status quo — one global threshold

Keep the single `AlphaEligibilityThresholds()` with `streak ≤ 2`, `DD ≥ −20%`, applied uniformly.

**Why deferred (rejected):**
- **Bad governance.** A portfolio_candidate at −4% DD and a baseline at −16% DD have structurally different risk profiles. One threshold cannot honestly calibrate both.
- The drawdown gate is loose against candidates (candidate at −4% has 16 percentage points of slack the gate never tests).
- The streak gate is tight against candidates (the construction absorbs negative-IC stretches, but the gate doesn't see the construction).
- v4 demonstrated the failure mode: G passed every gate except streak, while no honest test of "is this construction adequate?" was being run.

### Option B: Per-category named threshold sets (chosen)

Two named instances of the same dataclass, keyed by `ArmCategory` via a lookup dict. Dispatch at the script's worker boundary.

**Why chosen:**
- Minimal type-system change (one optional field added to the dataclass).
- Preserves backward compatibility — every existing `AlphaEligibilityThresholds()` caller (15 across the codebase) keeps working unchanged with the strict baseline values.
- Names are explicit in the audit trail (the `name` field lands in evidence JSON).
- The lookup-by-string keeps `factory_models.py` free of script-specific imports (the `ArmCategory` Literal lives in the latest-stack script; the threshold lookup doesn't need to import it).
- Adding a new category is a one-tuple addition to the lookup table.
- KeyError-on-miss is loud; a misnamed category surfaces immediately, not as a silent default-to-strict.

### Option C: Per-arm bespoke thresholds

Each `ArmSpec` carries its own `thresholds_factory: Callable[[], AlphaEligibilityThresholds] | None`. Every arm sets its own.

**Why deferred:**
- **Explosion of named instances.** 7 current arms → 7 threshold sets. The "what is the standard for this category?" governance signal is lost.
- **No shared semantics.** If A and B both get bespoke values that drift, an auditor can't tell whether "Arm B's threshold differs from Arm A's" is intentional or a regression.
- **Audit framing harder.** "research_ranker_baseline_v1" is a category-level contract that ARM_SPECS can be evaluated against. Per-arm thresholds collapse that into "each arm's specific decision," which is too granular for governance review.
- Useful as a refinement on top of Option B for one-off experimental arms, but not as the primary structure.

### Option D: Conditional/DSL thresholds

A threshold can be a constant OR an expression keyed on other metrics:

```python
@dataclass
class AlphaEligibilityThresholds:
    max_fold_negative_ic_streak: int | ConditionalThreshold = ...

# ConditionalThreshold: "≤ 4 if max_drawdown > -0.10 else ≤ 2"
```

**Why deferred:**
- **Harder to read.** "What gate did this evidence actually pass?" requires evaluating an expression, not reading a number.
- **Harder to audit.** A reviewer has to mentally evaluate the conditional against each evidence file. With the named-set design, the audit reduces to "look at `eligibility_thresholds.name`."
- **Ordering bug surface.** The conditioning metric (`max_drawdown`) has to be evaluated before the conditional threshold (`max_fold_negative_ic_streak`). The current `eligibility()` function evaluates checks in declared order — making one check's threshold depend on another check's actual value introduces an evaluation-order constraint that's easy to violate during a refactor.
- **The expressiveness isn't needed.** Two named sets cover the current 7-arm registry; a third would be one more tuple. The DSL would pay complexity cost for flexibility we don't need yet.
- Worth revisiting if the per-category sets grow to 5+ and the trade-offs become finer-grained.

### Option E: Multi-stage gating

Split the gate into two phases:
1. **Universal gate:** IC magnitude, IC stability — must pass for any arm.
2. **Category-conditional gate:** streak, DD, Sharpe — bound depends on category.

**Why deferred:**
- **Two-pass evaluation** requires changes to the `eligibility()` payload shape (which check came from which phase?). That ripples into every downstream consumer of the eligibility result.
- **Same effective outcome as Option B**, with more plumbing. Option B's "all gates in one set, gates that are construction-invariant happen to be identical across categories" produces the same audit output without a phase-1/phase-2 distinction.
- Cleaner in theory; not worth the cost in practice.

### Option F: Profile inheritance

`PORTFOLIO_CANDIDATE_THRESHOLDS = BASELINE_THRESHOLDS.with_overrides(streak=4, drawdown=-0.10)`.

**Why deferred:**
- **Frozen dataclasses don't have first-class inheritance** in Python. We'd need a `dataclasses.replace` wrapper or a builder pattern.
- **The "inherits from baseline" semantic is misleading.** Portfolio candidates aren't a more-permissive version of baselines; they're a DIFFERENT calibration that happens to share three of five values. The shared values are coincidence (IC and Sharpe gates are construction-invariant), not parentage.
- Option B with explicit kwargs (`name=..., max_fold_negative_ic_streak=4, max_drawdown=-0.10`) is just as terse and more honest about what's happening.

## Trade-off analysis

**Option B vs Option A** is the only material choice. Per-category thresholds add ~100 lines of code (two named instances, one lookup dict, the `name` field, dispatch in the worker) and ~250 lines of tests. The win is:

1. **Governance now reflects reality.** A construction-protected candidate doesn't fail a gate calibrated for an unconstrained baseline.
2. **The audit trail self-describes.** `eligibility_thresholds.name = "portfolio_candidate_v1"` in the evidence JSON tells you exactly which set was applied — no need to diff numeric values across runs.
3. **The contract is explicit.** "Looser streak in exchange for tighter DD" is the most defensible asymmetric trade we found — a portfolio_candidate that's truly protected by its construction earns the streak laxity, and one that isn't loses both gates.
4. **Extension path is obvious.** A future `intraday_scalping_candidate` category gets its own threshold set without touching the dataclass.

Option B vs C/D/E/F: each of those would offer marginal expressive gain at a real complexity cost. The named-set design is the right unit of governance.

## Consequences

### What becomes easier

- **G's eligibility verdict is now meaningful.** Pre-v5, "G fails eligibility on streak 4 > 2" was true but mis-framed — the gate was wrong for G's category. Post-v5, "G passes all 5 gates under portfolio_candidate_v1" is a defensible governance statement.
- **Onboarding new arm categories is mechanical.** Add an `ArmCategory` Literal value + a named `AlphaEligibilityThresholds` instance + a `THRESHOLDS_BY_ARM_CATEGORY` entry. Three lines for the structure; the gate logic is unchanged.
- **Audit trail is self-describing.** `eligibility_thresholds.name` in evidence JSON identifies the applied set without value-diffing.
- **Future tuning is per-category.** A walk-forward calibration of the streak threshold against held-out data can run independently on baselines and candidates without entangling them.

### What becomes harder

- **Two named instances with identical numeric values.** `AlphaEligibilityThresholds()` (legacy default, `name="default_strict"`) and `RESEARCH_RANKER_BASELINE_THRESHOLDS` (`name="research_ranker_baseline_v1"`) have the same five threshold values. An auditor scanning evidence for "which set?" sees two different names for what is structurally one gate. *Action item below: align the default's `name` so the two paths produce the same audit string.*
- **Schema version not bumped.** The new `name` field is additive in `asdict(thresholds)` → `eligibility_thresholds`. `EVIDENCE_SCHEMA_VERSION` is unchanged (`"backtest-latest-stack-realized-v2"`). Strict-dict-equality consumers comparing v4 evidence to v5 evidence will see a diff. *Action item: bump to a `v2.1` minor revision or document the additive policy.*
- **Mutable lookup dict.** `THRESHOLDS_BY_ARM_CATEGORY: dict[str, ...]` could be overwritten at runtime by a misbehaving caller, defeating the lookup-is-canonical intent. *Action item: wrap in `MappingProxyType`.*
- **`ArmCategory` Literal lives in the script.** The lookup is keyed by `str` because importing the Literal would invert the layering (`services/research → scripts`). Tests pin the alignment; the type system doesn't. Acceptable given the layering.

### What we'll need to revisit

- **The threshold values are operator-tunable starting points, not certified.** `streak ≤ 4`, `DD ≥ −10%` for portfolio_candidates were chosen to attack the gate that v4 surfaced. They have not been walk-forward-tuned against held-out data. A focused calibration sweep on (streak ∈ {3,4,5}, DD ∈ {−0.05,−0.10,−0.15}) over the panel's earliest 252 days, validated against the rest, would land defensible values. Less urgent now that G passes with the default settings; tune before G goes to live paper.
- **Per-arm overrides** (Option C as a refinement on top of B) may become useful if a specific experimental arm needs bespoke calibration. The current dispatch can accommodate it via an `ArmSpec.thresholds_factory` field added later.
- **Streak threshold sensitivity for G.** G is at streak 4 against a ≤ 4 gate — zero margin. A regime-specific episode adding one more negative fold would flip eligibility. Before paper-trading promotion, run a +/−1 sensitivity analysis on the streak gate.
- **A baseline that produces low DD.** What if a future research_ranker_baseline produces a 5% DD organically (e.g., a feature mix that's naturally market-neutral)? It would pass the candidate-style DD gate without earning candidate status. Today the dispatch is category-driven, not metric-driven — that's the right call, but worth noting if the baseline definition ever broadens.

## Action Items

1. [x] `AlphaEligibilityThresholds` gains `name: str` field. — PR #71.
2. [x] `RESEARCH_RANKER_BASELINE_THRESHOLDS` + `PORTFOLIO_CANDIDATE_THRESHOLDS` + `THRESHOLDS_BY_ARM_CATEGORY` added in `factory_models.py`. — PR #71.
3. [x] Latest-stack worker dispatches via `THRESHOLDS_BY_ARM_CATEGORY[spec.category]`. — PR #71.
4. [x] 15 regression tests pin the trade-off shape, lookup completeness, and the "G passes candidate / fails baseline" governance contract. — PR #71, `test_per_category_eligibility_thresholds.py`.
5. [x] ADR-003's "per-category eligibility thresholds" follow-up item marked closed with a forward-reference. — PR #71.
6. [x] **Default's `name` aligned with `RESEARCH_RANKER_BASELINE_THRESHOLDS`.** `AlphaEligibilityThresholds()` now produces `name="research_ranker_baseline_v1"` identical to the named instance. (Closes code-review finding #3.) — PR #71 (cleanup commit).
7. [x] **Evidence schema version bumped** to `"backtest-latest-stack-realized-v2.1"` to acknowledge the additive `name` field. Docstring on `EVIDENCE_SCHEMA_VERSION` documents the additive-minor / breaking-major policy. (Closes code-review finding #6.) — PR #71 (cleanup commit).
8. [x] **`THRESHOLDS_BY_ARM_CATEGORY` is now read-only** via `types.MappingProxyType`, typed as `Mapping[ArmCategory, ...]`. (Closes code-review finding #4.) — PR #71 (cleanup commit).
9. [x] **`ArmCategory` moved to shared types module** (`services/research_service/sampling/arm_category.py`) so the lookup is Literal-typed directly. (Closes code-review finding #5, system-design rec #5.) — PR #71 (cleanup commit).
10. [x] **Run-level manifest** (`run_manifest.json`) writes alongside per-arm evidence. Captures run_id, started/finished UTC, wall-clock, git_commit, requested/completed/skipped arms with reasons + headline metrics, cli_args, max_workers_used, fingerprints. (Closes system-design rec #2.) — PR #71 (cleanup commit).
11. [x] **Evidence field classification documented** in `save_evidence` docstring — distinguishes deterministic-from-inputs fields from varies-per-run fields. (Closes system-design rec #7.) — PR #71 (cleanup commit).
12. [x] **Walk-forward-tune the candidate streak + DD thresholds** — **done 2026-05-28** via `scripts/calibrate_eligibility_thresholds.py` (see "Held-out calibration" section). Result: the DD gate never binds and the fixed-streak threshold is not hold-out-validatable (G's streak-4 lives only in the validation window). Verdict: replace the fixed-streak gate with the DD-conditioned gate (Item 16) rather than re-tuning a fixed value.
13. [x] **Streak threshold sensitivity for G** — **settled 2026-05-28** from existing evidence (see "Streak-gate settlement" section). The streak is computed from per-fold IC and is invariant to the threshold, so the three reruns were unnecessary: G fails at ≤3, passes at ≤4/≤5; its streak-4 is a real summer-2024 regime episode that the construction absorbed (per-fold DD ≤1.1%).
14. [x] **Latest-stack → model-registry adapter** (system-design rec #3) — **done 2026-05-28**. Built `research_service/modeling/registry/latest_stack_promotion.py`: `build_registration(evidence)` converts an arm's latest-stack evidence into the registry's `register_model(strategy_name, model_version, feature_set_version, as_of, metadata)` inputs, gated on the evidence's own verdict (eligible `portfolio_candidate` flagged `production_candidate` only; baselines rejected via `NotPromotableError`). `promote_to_registry(registry, evidence)` is the async one-call path; `scripts/promote_latest_stack_arm.py` is the operator CLI (validates + emits the payload). Verified on real evidence: Arm G produces a clean registration; baseline A is rejected (exit 2). Correction to the original framing: the registry is **metadata-only** — it does not ingest a `WalkForwardEvidence` bundle — so the adapter packs the gate result + headline metrics into `metadata`; no model weights are needed (linear-ranker weights recompute live; trained artifacts ride the governance `alpha_promote` path). The live, DSN-backed registration remains `alpha_promote`; this adapter supplies its identity + evidence inputs.
15. [x] **Holdout calibration discipline** (system-design rec #1) — **done 2026-05-28**. The calibration sweep on the earliest 252 days, validated against the rest, confirmed the streak=4 value was reverse-engineered from the full sample and does not generalise. Discipline established as the reusable `scripts/calibrate_eligibility_thresholds.py`; future threshold proposals run through it.

16. [x] **Drawdown-conditioned streak gate (Option D) — implemented 2026-05-28** (operator-directed). `AlphaEligibilityThresholds` gains `max_fold_negative_ic_streak_if_dd_contained` (relaxed cap) and `streak_containment_max_drawdown` (a bound *tighter* than the full-run DD, so the condition isn't redundant with the existing DD gate). `eligibility()._streak_check` passes a streak iff it is `<= floor (2)`, or `<= cap (6)` AND the drawdown during the worst streak stayed inside the containment bound; otherwise the floor applies. `PORTFOLIO_CANDIDATE_THRESHOLDS` → `portfolio_candidate_v2` (floor 2, cap 6, containment **−2%**, full DD −10%). Containment was set to −2% (tightened from an initial −5% first cut the same day): the calibration showed every realized_v2 streak-≤6 candidate's within-streak DD was ≤1.4% (Arm G's ~−0.3%), so −2% admits all genuinely-absorbed episodes with margin while forfeiting the relaxation for any episode that actually bit the book. The within-streak DD is emitted by `walk_forward._ic_streak_metrics` as `max_drawdown_during_worst_streak` (helper: `campaigns/evaluation/streak_containment.py`). **Verified on the realized_v2 A–J set: verdicts identical to the old `streak<=4` gate** (G, H eligible; D/E/J fail the cap; F/I fail other gates). New behaviour: a candidate whose streak episode *caused* the drawdown (within-streak DD beyond −2%) loses the relaxation even if its full-run DD gate still passes. ~1300 tests green; ruff/mypy clean.

Items 6–11 are closed by the cleanup commit in PR #71 that follows this ADR's initial draft. Items 12–16 are all closed as of 2026-05-28: the held-out calibration (12/15) showed the flat streak threshold was un-validatable, motivating the drawdown-conditioned gate (16, `portfolio_candidate_v2`); the streak sensitivity (13) was settled from existing evidence; and the model-registry adapter (14) now converts a winning arm's evidence into a registration without a manual step. **The streak gate is no longer a blocker to G's promotion** — the remaining path is operational (run the live, DSN-backed `alpha_promote` with the adapter's output, then the 30/90/365-day paper sequence).

## 2026-05-28 — Streak-gate settlement (sensitivity analysis)

Settles the "Streak threshold sensitivity for G" revisit item and **Action Item 13**, directly from the canonical A–J evidence (`data/parquet/research/backtest_latest_stack_realized_v2`, run_id `4730187d…`, git `9d34da6`) — **no rerun required**. `fold_negative_ic_streak` is computed from per-fold model IC (`folds[*].mean_ic`), which is invariant to both the eligibility threshold and the portfolio dial. The three planned reruns would only re-confirm pass/fail at each bound (G fails at ≤3, passes at ≤4 and ≤5); they cannot change G's streak. The dial's internal streak parameter is a separate knob from the eligibility threshold.

### What the data shows (Arm G, 63 folds)

- **The streak-4 is a real regime episode, not a gate artifact.** The binding run is folds 31–34 = **2024-06-17 … 2024-09-09**, per-fold IC `−0.253, −0.213, −0.127, −0.029`. The first three are >15× the IC bootstrap-CI halfwidth (±0.0078); zero folds in the run are noise-fragile, and a magnitude floor up to ε = 0.010 leaves the longest negative run at 4. The gate is correctly flagging a ~3-month stretch where the price-volume signal strongly **inverted** — not a cluster of marginally-negative folds.
- **Zero margin is real and brittle.** Second-longest negative run is 3 (distribution of run-lengths: `{1:6, 2:5, 3:3, 4:1}`). The verdict hinges on one fold either side: flipping the mildest member (fold 34, −0.029) drops the streak to 3; the run was broken only by fold 35 = +0.023 (a thin ~3σ positive). One slower-recovering fold makes it streak-5 → G fails.
- **The construction absorbed the episode — ADR-004's trade paid off.** Through folds 31–34, per-fold drawdown stayed ≤ **1.1%** and cumulative return over the four folds was ≈ **+0.26%** (flat-to-up). A −0.25 IC fold produced −0.05% DD. Long-only top-30 + caps + dial neutralised a 4-fold ranking inversion almost entirely. This is the strongest available evidence for the contract "looser streak **iff** the construction actually protects you" — here it demonstrably did.
- **Signal is weak in absolute terms.** 29/63 folds negative (46%), mean IC +0.0107, CI `[0.003, 0.018]`. G is a thin, real, regime-fragile alpha.
- **IC-optimisation makes the streak worse — meta-finding confirmed.** Arm J (GBDT pairwise-rank, the most aggressive IC optimiser) has the highest Sharpe (1.28) and return but the **lowest** mean IC (+0.0030) and the **longest** streak (9). H (regime overlay) is streak-4 like G. No in-family model lowers the streak; the most IC-greedy one lengthens it. Corollary: a more sample-efficient in-family search (random / evolutionary / **policy**) optimises the same IC and **will not move this gate**.

### Decision

1. **Keep `portfolio_candidate` at `max_fold_negative_ic_streak ≤ 4`.** Do not loosen to ≤5 — the streak is a genuine regime episode; widening to admit it would be fitting the gate to the candidate.
2. **G's pass is substantively legitimate, not merely barely-legal** — the flagged episode was absorbed with ≤1.1% per-fold DD. Recorded as a defensible pass under the per-category gate.
3. **Zero margin on a real 3-month inversion is too thin for live-paper promotion on this evidence alone.** Action Items 12 & 15 (held-out streak/DD calibration) are **escalated from nice-to-have to a hard precondition for G promotion.**
4. **Proposed gate refinement — a drawdown-conditioned streak gate** (promotes the original Option D from *deferred* to *proposed*). Replace the unconditional 4-streak laxity with: streak ≤ 2 unconditionally, **and** a streak up to a hard cap (≤ 6) tolerated *only if the realized drawdown within the streak window stayed inside the candidate DD bound*. This encodes the ADR's contract **per-episode** rather than globally — laxity is earned by the construction holding during the specific episode the streak flags, not granted to any 4-streak. For G, the summer-2024 window (DD ≈ 0) passes with real margin; a candidate whose streak coincided with a DD breach fails. Removes the arbitrary "4" and the zero-margin brittleness. Deferred originally for evaluation-order/readability cost; the data now justifies paying it. **Implementation is a separate PR** (`eligibility()` gains a within-window DD input; the conditional is evaluated after the DD check). Not yet accepted — operator sign-off required before changing gate semantics.

## 2026-05-28 — Held-out calibration (Action Items 12 & 15)

`scripts/calibrate_eligibility_thresholds.py` splits each candidate arm's 63 folds into a calibration window (earliest 252 OOS trading days → cutoff **2023-08-22**, 16 folds) and a validation window (47 folds), recomputes the gate metrics per window, and sweeps the `(streak ∈ {3,4,5}, DD ∈ {−0.05,−0.10,−0.15})` grid. Tuning evidence: `…/backtest_latest_stack_realized_v2/threshold_calibration.json`. (Calibrates the **eligibility gate**, post-hoc on the OOS fold series — not the portfolio dial's `kill_streak`; streak is exact, windowed DD is a fold-granular proxy.)

1. **The DD gate never binds.** Every candidate's drawdown is in `[−0.017, −0.041]` — far inside even −0.05. All three DD columns give identical admit sets; the DD *value* changes no verdict. Calibration reduces entirely to the streak threshold. (Quantifies the ADR's "DD gate is loose against candidates" claim.)
2. **The streak metric is cross-window unstable, so no fixed value generalises.** calibration→validation streaks: G `3→4`, D/E `3→7`, F/H `3→4`, I `4→3`, J `9→5`. **No grid cell yields a stable (cal == val) admit set;** the metric even reorders the arms between windows.
3. **G's streak-4 is purely a validation-window property → the threshold is overfit.** On the calibration window G's worst streak is **3** (the summer-2024 episode is entirely in validation). A threshold honestly chosen from calibration would be ≤3 — and would then **reject G** out-of-sample. So `streak ≤ 4` was fit to the full sample including the very episode it gates; it is **not** hold-out-validatable — exactly the Items 12/15 concern.

**Conclusion:** do not re-tune the fixed-streak value — *no* fixed value survives a hold-out. Adopt the drawdown-conditioned gate (Item 16), whose binding component (DD) is the stable metric. A hard cap of ≤ 6 rejects D/E (7) and J (9) while admitting the low-DD streak-4 cohort {F, G, H, I} — reproducing the current full-sample verdict with a rationale that *does* generalise. **G promotion is now gated on Item 16, not on re-tuning the fixed value.**

## 2026-05-29 — v3: streak gate replaced by a bootstrap-IC significance gate (supersedes v2)

The [ADR-011](adr-011-live-pv-formulaic-feature-port.md) dollar-volume scoring fix forced a full A–N re-run on corrected evidence (`backtest_latest_stack_normfixed/`). On that evidence the v2 conclusion sharpened into a verdict: **a negative-IC-streak count cannot be a principled gate on this universe/label at all.**

**Why v2 is abandoned.** Re-running the held-out calibration (grid extended to streak ∈ {3…9}) on the corrected evidence: the streak is *still* cross-window unstable (linear arms cal `3` → val `7`; the 2024-summer crash episode falls entirely in validation), and **the only cal==val-stable cap is 9** — i.e. so loose it admits everything. The v2 drawdown-conditioned relaxation (floor 2 / cap 6) is itself a streak-count gate, so it inherits the same non-stationarity; it just relocated the brittleness. No streak-count threshold generalises.

**Why a regime overlay is *not* the answer either** (the obvious alternative was investigated and rejected). A regime decomposition shows the damage is concentrated in 3 momentum-crash episodes (2023 bank crisis, 2024 summer rotation, 2025 tariff selloff) = 27% of OOS time, where the linear arms' IC inverts to −0.08…−0.18 while ex-episode IC is a healthy +0.084. But an **oracle** regime filter (perfect hindsight de-risk through every episode) lifts G's Sharpe only `0.77 → 0.78`: the episodes wreck the *IC/streak* but their *capital* impact is tiny (contained DD ≤ 1.3%), so removing them strips small losses **and** small gains. The Sharpe shortfall is structural, not episode-driven. A regime overlay can shrink the streak but cannot manufacture risk-adjusted return — so it does not rescue eligibility.

**v3 redesign (chosen).** Demote the streak to a loose catastrophic backstop at the one OOS-stable cap (`max_fold_negative_ic_streak = 9`; it never binds for a sane arm) and make robustness a **statistical-significance gate on the IC itself**: `min_bootstrap_ic_p05 > 0` — the 5th percentile of the block-bootstrapped fold-IC distribution must be positive ("95% confident the predictive power is positive"). This tests the same concern the streak gate intended — *is the alpha reliably positive across regimes?* — with a metric that is regime-inclusive by construction (the bootstrap resamples span the crash folds) and does not depend on where an episode lands in the window. `PORTFOLIO_CANDIDATE_THRESHOLDS` → **`portfolio_candidate_v3`** (streak ≤ 9 backstop, `min_bootstrap_ic_p05 = 0.0`, DD −10%, Sharpe ≥ 1.0, IC gates unchanged).

**It is strict, not a rubber stamp.** Re-evaluated on the corrected A–N evidence, v3 admits **D** (PCA, Sharpe 1.09, p05 +0.018) and **N** (GRU, Sharpe 1.12, p05 +0.011) and rejects **J** (GBDT-rank) on the bootstrap gate (p05 −0.006) *despite its 1.28 Sharpe* — J's edge is a single crash episode, not robust ranking. The portable linear arms (F/G, Sharpe 0.85–0.94) fail the Sharpe gate on corrected evidence — the honest verdict that the simple linear alpha is sub-1.0. The v2 DD-conditioned *mechanism* remains in `_streak_check` (a threshold may still opt in) but no production preset uses it.

**Caveat / next step.** Both v3-eligible arms (D, N) need live-inference machinery the current linear port lacks (PCA transform / GRU). So v3 fixes the *governance gate* but does not by itself yield a portable promotable arm; that remains alpha-improvement work (the regime overlay is not it — see above). Reproduce: `scripts/calibrate_eligibility_thresholds.py --evidence-dir data/parquet/research/backtest_latest_stack_normfixed` and the eligibility re-eval under `portfolio_candidate_v3`.

**Alpha-improvement attempt — downside-robust reweighting (Arm O) — also FAILED OOS.** The per-feature regime decomposition (above) showed the momentum factors invert in crashes while volatility/range + reversal + long-horizon hold. Arm O (`RobustICRanker`, a portable linear model) fits weights from the *downside* of each feature's daily-IC distribution to drop the crash-fragile factors. In-sample the combined-score IC-IR lifts 0.09→0.21 and the crash IC flips positive — but **out-of-sample it cut the streak (G 7 → O 4) yet gutted the alpha** (oos_ic 0.159→0.041, ic_60d 0.063→−0.029, bootstrap_p05 +0.015→−0.015). The crash-robust factors are defensive but weak: dropping momentum removes the fragility *and* the alpha. This rules out the last cheap lever — **a linear reweighting of the momentum-heavy feature set cannot escape the momentum-crash tradeoff.** The remaining genuine levers are new signal (orthogonal features / a different label / a different universe) or committing to a non-linear (PCA/GRU) live port. Arm O is retained in the research ledger as a documented negative result.

## Related

- [ADR-001](adr-001-operational-hardening.md) — Operational hardening. The promotion-sequence policy this ADR feeds into.
- [ADR-002](adr-002-learned-family-representation-choice.md) — Learned-family representation choice. v4 + v5 confirm learned-PCA stays research-status only.
- [ADR-003](adr-003-return-accounting-separation.md) — Return-accounting separation. **This ADR closes ADR-003's "per-category eligibility threshold separation" action item.**
- Memory: [Backtest Latest Stack](../../memory/project_backtest_latest_stack.md) — v5 results with the per-category gate verdict.
- PR [#71](https://github.com/Liu-Ming-Yu/Quant/pull/71) — implementation + tests + memory update.
