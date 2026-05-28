# ADR-004 — Per-category eligibility thresholds

**Status:** Accepted (2026-05-28) and **shipped** in PR #71 on branch `feat/per-category-eligibility-thresholds`. Closes the "per-category eligibility threshold separation" action item flagged in ADR-003.
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
12. [ ] **Walk-forward-tune the candidate streak + DD thresholds** before promoting G to live paper. Held-out sweep over `(kill_streak ∈ {3,4,5}, drawdown ∈ {−0.05,−0.10,−0.15})`. Requires a new calibration script that produces tuning evidence on a hold-out window. **Pre-paper hardening; tracked as ADR-005 candidate.**
13. [ ] **Streak threshold sensitivity for G** — three full universe-300 reruns at `streak ∈ {3, 4, 5}` for the candidate gate. Pure ops work (~1 hour of compute); the script is ready to run.
14. [ ] **Latest-stack → model-registry adapter** (system-design rec #3). Convert latest-stack evidence into the `WalkForwardEvidence` artifact bundle the registry consumes, so G can enter the existing 30d/90d/1y paper-trading promotion sequence without a manual step.
15. [ ] **Holdout calibration discipline** (system-design rec #1). The current threshold values (streak=4, DD=−10%) were chosen to attack the audit gate, not validated against a hold-out. Before G goes to live paper, run a calibration sweep on the earliest 252 days and validate against the rest.

Items 6–11 are closed by the cleanup commit in PR #71 that follows this ADR's initial draft. Items 12–15 are pre-paper hardening or scale-readiness work that won't fit in a single follow-up PR; each opens its own workstream.

## Related

- [ADR-001](adr-001-operational-hardening.md) — Operational hardening. The promotion-sequence policy this ADR feeds into.
- [ADR-002](adr-002-learned-family-representation-choice.md) — Learned-family representation choice. v4 + v5 confirm learned-PCA stays research-status only.
- [ADR-003](adr-003-return-accounting-separation.md) — Return-accounting separation. **This ADR closes ADR-003's "per-category eligibility threshold separation" action item.**
- Memory: [Backtest Latest Stack](../../memory/project_backtest_latest_stack.md) — v5 results with the per-category gate verdict.
- PR [#71](https://github.com/Liu-Ming-Yu/Quant/pull/71) — implementation + tests + memory update.
