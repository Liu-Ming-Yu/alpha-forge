# ADR-003 — Return-accounting separation: label vs realized P&L

**Status:** Accepted (2026-05-26) and **shipped** — implementation merged in PR #62 (2026-05-27) across four commits on branch `fix-return-accounting-and-purge`. Six of the eight action items below are closed; the two remaining (ADR-004 trading-day fold generator, mining-driver opt-in) are deferred follow-ups.
**Context:** Audit fix that closed the v1-`backtest_latest_stack` bug in which a 21d forward log-return label was compounded as a 1d realized P&L, inflating total return by ~21× per fold horizon (e.g. Arm A reported total_return ≈ 38,492× on universe-300).

## Context

`SupervisedAlphaSample` historically carried a single return field — `forward_return` — used as both the **predictive label** (input to IC, feature weighting, bootstrap CIs) *and* the **portfolio P&L source** (input to Sharpe, max drawdown, total return, equity curve). Two evaluators — the signed-rank `daily_metrics` and the long-only `evaluate_long_only_portfolio` — fed `forward_return` directly into a daily-cadence return series and compounded each value via `equity *= 1 + value` and annualized Sharpe with `sqrt(252)`.

When `forward_return` represents a 1d return (e.g. early-stage research with `horizon_days=1`), this happens to be correct. When `forward_return` represents a 21d log return (the current standard for the latest stack), it is **not** a 1d realized P&L: it conflates a multi-day horizon with daily compounding cadence, double-counts overlapping windows when daily samples are emitted, and inflates returns by a factor that scales with the horizon.

The audit raised this in five connected concerns that all bear on the sample model:

1. The compounding contract (`equity *= 1 + r`) demands **simple** returns; a switch to log returns is a subtler version of the same bug.
2. The purge gap (`purge_days = 5`) was shorter than the label horizon (21) — even a structurally correct return field would have label leakage.
3. The signed-rank arms have no per-name caps, sector neutralization, ADV caps, borrow model, or cash model — they are diagnostic baselines, not portfolio candidates.
4. The evidence schema (`backtest_latest_stack_v1`) carries pre-fix and post-fix evidence under the same semantic version, so dashboards silently compare incompatible payloads.
5. The downstream evidence consumers (artifact payloads, tearsheet, eligibility gate) all read `evidence.metrics` and `evidence.daily_returns` as the canonical return surface — any fix must keep these consumers working.

This ADR records the architectural choices made to fix the bug while preserving the existing evaluator API, the existing test suite, and the existing evidence consumer contracts.

## Decision

The `SupervisedAlphaSample` contract gains an explicit return-accounting separation, and the evaluators switch their P&L source on detection of the new fields.

### 1. Two return fields on the sample, not one

`SupervisedAlphaSample` now carries:

* `forward_return: float` — **label only**. The multi-day forward log return (e.g. `log(close[t+21]/close[t])` for `horizon_days=21`). Consumed by IC, feature weighting, fold mean IC, bootstrap IC, decile-spread metrics. Never compounded as a daily realized P&L.
* `realized_return_1d: float | None = None` — **P&L only**. The one-day **simple** realized return `close[t+1] / close[t] - 1.0`. Consumed by Sharpe, max drawdown, total return, equity curve. Compounded via `equity *= 1 + r` — simple, not log, so the existing compounding contract is correct without a parallel codepath.
* `as_of_index: int | None = None`, `label_end_index: int | None = None`, `label_end_as_of: datetime | None = None` — global-trading-day-calendar positions used by the sample-level purge.

The three new fields default to `None` so existing JSON artifacts, existing test fixtures, and existing callers (model registry, paper-soak, mining) keep working without change. The serializer drops unset optional fields; the loader treats absence as `None`.

### 2. Runtime opt-in via field-presence detection

The evaluators detect realized mode at the top of each call:

```python
realized_mode = all(row.realized_return_1d is not None for row, _ in scored)
```

When all samples carry `realized_return_1d`, the evaluator runs in realized mode (rebalance once per fold, hold weights, mark-to-market with 1d simple returns). When any sample lacks it, the evaluator falls back to **legacy mode** — the pre-fix behavior — with a docstring warning that legacy mode "carries the well-known label-as-P&L bug and must not gate production decisions."

### 3. Sample-level purge via global trading-day indices

`WalkForwardConfig.label_horizon_days: int | None = None` is added with `__post_init__` validation that enforces `purge_days >= label_horizon_days` *only when set*. The latest-stack driver sets it; existing callers don't and keep the old `purge_days >= 0` validation.

Inside `run_sample_walk_forward`, when every sample in the fold's working set carries `as_of_index` and `label_end_index`, a sample-level filter drops training rows whose `label_end_index >= min(test.as_of_index)`. The selected mode is recorded per-fold under `fold_basis ∈ {"calendar_days", "calendar_days_plus_sample_label_index_purge"}` so the evidence is self-describing.

### 4. Bucket-21d series derived from the daily MtM stream

A diagnostic non-overlapping 21d-bucket return series is computed by compounding the *canonical daily MtM stream*, not by a parallel engine:

```python
bucket_returns = non_overlapping_bucket_returns(daily_mtm, horizon_days=21)
bucket_sharpe(bucket_returns, horizon_days=21)  # annualized with sqrt(252/21)
```

The daily MtM stream is the **single source of truth**; bucket is a derived view. Both are emitted to the evidence JSON (`metrics_daily_mtm` and `metrics_bucket_21d`), but only daily MtM feeds the eligibility gate.

### 5. Versioned evidence schema with both root and field-level bumps

* Output root: `data/parquet/research/backtest_latest_stack_v1/` → `data/parquet/research/backtest_latest_stack_realized_v1/`. Old v1 evidence stays on disk for diff and is never overwritten.
* Evidence JSON gains `evidence_schema_version = "backtest-latest-stack-realized-v1"` as a top-level field.

Both, not one. The root bump prevents accidental dashboard mixing; the version field lets a single-file consumer self-check.

### 6. Existing `evidence.metrics` semantics preserved

`evidence.metrics`, `evidence.daily_returns`, and `evidence.daily_turnover` retain their existing meaning (the gated daily MtM stream). The new fields are emitted alongside them in the JSON payload as `metrics_daily_mtm` (alias) and `metrics_bucket_21d` (additive). Downstream consumers — `walk_forward_artifact_payloads`, the tearsheet, the eligibility gate — continue to work without change.

### 7. CLI rename with backward-compatible aliases

Arms get canonical names (`research_ranker_pv`, `research_ranker_pv_formulaic`, `research_ranker_pv_formulaic_learnedpca`, `long_only_top30_pv_formulaic_learnedpca`) plus `arm_category` and `production_candidate` tags. The CLI aliases `A,B,C,D` continue to resolve to the new canonical names so saved scripts and operator muscle memory don't break.

## Alternatives considered (and why deferred)

### Single semantic-overload field

Keep `forward_return` as the only return field; switch its semantic from "label" to "1d simple realized return" via a `return_type` discriminator (which already exists as an unused enum on the dataclass).

**Why deferred:**

- Loses the IC signal. A 1d simple return is a much noisier IC label than a 21d log return; rank-IC is a directional statistic whose power benefits substantially from longer horizons. The IC, feature weighting, and bootstrap CI machinery was tuned against the 21d label.
- Confuses the bug being fixed. The bug isn't "we used the wrong field name" — it's "we used one field for two semantically incompatible roles." Switching the role doesn't separate the roles; the next maintainer would just re-encounter the same confusion in a different direction.
- The serialization contract becomes mode-dependent (the float means different things depending on `return_type`), which complicates round-tripping samples through JSON and through the model registry.

### Parallel realized-return panel keyed by (instrument_id, as_of)

Build a separate `RealizedReturnPanel` DTO carried alongside `SupervisedAlphaSample` and joined inside the evaluator.

**Why deferred:**

- Doubles the data plumbing surface. Every caller that constructs samples (latest-stack driver, mining driver, paper-trade-replay, fold splitters) would also need to construct, pass, and serialize a panel. Today a sample is self-contained; that property has paid off repeatedly during refactors.
- Breaks per-sample provenance. A sample is the atomic unit for governance — promotion gates, attribution, audit trails all key on individual samples. A side-channel realized return that lives "near" but not "in" the sample makes governance harder.
- No real benefit. The realized return is fully determined by the same `(instrument_id, as_of)` that already identifies the sample; collocating the two fields is just the cheaper layout.

### Required `realized_return_1d` (no optional default)

Make the new field mandatory.

**Why deferred:**

- Breaks every existing test fixture and every existing sample JSON. The model registry has paper-soak runs going back weeks, all of which carry samples without the new field. Forcing migration of all of them is a sprint's worth of work — the optional-field path is a 30-line patch.
- Doesn't add any safety the runtime check doesn't already provide. If a caller forgets to populate the field, today the evaluator falls back to legacy mode (with an explicit docstring warning); making the field required only changes the error from "subtle wrong number" to "loud crash" — but the runtime check + the `production_candidate: false` tagging on legacy-prone arms catches the same class of error.

### Explicit mode parameter on the evaluators

Add `mode: Literal["realized", "legacy"]` to `daily_metrics` and `evaluate_long_only_portfolio`.

**Why deferred:**

- Pushes the mode decision into every caller. Today the mode is fully determined by the data; pushing it to the caller adds a parameter that can be set inconsistently with the data and silently corrupt the run.
- Doesn't compose. The `WalkForwardConfig.label_horizon_days`, the sample-level purge, the `realized_return_1d` field, and the evaluator mode are all *the same decision*: "this fold's samples carry realized accounting." Splitting that decision into three or four independent boolean toggles is exactly the kind of state-explosion the runtime check avoids.

### Switch fold generator to trading-day index

Replace `generate_folds` calendar-day boundaries with trading-day-index boundaries so the purge gap is directly expressible in trading days.

**Why deferred:**

- Touches every walk-forward caller, not just the latest-stack one. The fold generator is used by the model registry, the mining walk-forward, the paper-trade-replay, the campaign portfolio backtest, and every test that constructs folds — a substantial refactor.
- The sample-level purge captures most of the safety benefit. Once each sample knows its global trading-day position and its label-end position, the purge can be enforced row-by-row in the driver, independent of how folds are generated. The calendar fold generator remains the broad cut; the sample filter is the precise cut.
- Worth doing eventually, but as a focused ADR-004, not as part of the return-accounting fix.

### Trading-day-aware schema break (rename `forward_return` → `label_return`)

Bump `SupervisedAlphaSample` schema; rename `forward_return` to `label_return`; add `realized_return_1d` as required.

**Why deferred:**

- Breaks the existing model registry, every sample JSON on disk, every promotion artifact, and every paper-soak archive. The schema is consumed by all five services.
- Doesn't actually solve a problem the additive change doesn't. The name `forward_return` is accurate (it *is* a forward return), and adding `realized_return_1d` next to it makes the role separation explicit at the API surface.

## Consequences

### What becomes easier

- **Audit reproducibility.** Each evidence file now carries `evidence_schema_version`, `git_commit`, `universe_fingerprint`, `bars_snapshot_fingerprint`, `feature_set_versions`, `eligibility_thresholds`, `walk_forward_config`, `label_horizon_days`, `return_mode_daily`, `return_mode_bucket`, `realized_mode_used`, `fold_basis`, `cli_args`, and `n_folds_actual`. A future auditor can answer "what produced this number, against which data, under which gates" without leaving the file.
- **Mode-incompatible runs no longer mix.** The output-root bump means dashboards can't accidentally compare pre-fix and post-fix metrics. The field-level `evidence_schema_version` lets a single-file viewer self-check.
- **The eligibility gate is now telling the truth.** Pre-fix, all four arms failed on inflated drawdown — informative direction, garbage magnitudes. Post-fix, Arm A actually passes the Sharpe gate (1.0546 ≥ 1.0); the binding constraint is the fold negative-IC streak. This unblocks meaningful regime-overlay / risk-dial work.
- **Adding new arms is mechanical.** `ARM_SPECS` is a tuple of dataclasses with `cli_alias`, `canonical_name`, `category`, `production_candidate`, `panel_key`, `requires_pca`, `portfolio_config_factory`. The dispatch loop is fully data-driven (no `if/elif` branch per arm); a new arm is a single tuple entry plus, if it's a portfolio arm, a factory function. The CLI parser accepts either alias or canonical name.

### What becomes harder

- **The sample model has more surface area.** Three new optional fields plus updated serialization. New callers must remember to populate the indices and the realized return if they want governed accounting. Mitigated by: (a) the latest-stack builder is the only current caller, (b) the runtime check loudly signals legacy fallback in the docstring, (c) the all-or-nothing validation in `run_sample_walk_forward` rejects mixed inputs loudly, (d) the shared `has_realized_returns(scored)` helper colocates the contract with `SupervisedAlphaSample`.
- **The bucket-21d series is "derived but separately rendered"** — dashboards built against `metrics_bucket_21d` could drift from the daily MtM stream if a future change updates one but not the other. Mitigated by: derivation is one function call in `save_evidence`, not a parallel engine, so accidental drift requires actively breaking the derivation.

### What we'll need to revisit

- **Trading-day fold generator** (ADR-004 candidate): the sample-level purge fixes the leakage at the row level, but the fold *boundaries* are still calendar-day. For a 21-trading-day test_window, the boundary lands ~31 calendar days later, which is wider than needed. Move when the cost of touching every caller is paid for by a concrete benefit.
- ~~**Eligibility-gate threshold separation per arm category**~~ — **closed by PR #71** (2026-05-28). `factory_models.py` now exposes `RESEARCH_RANKER_BASELINE_THRESHOLDS` (`max_fold_negative_ic_streak=2`, `max_drawdown=-0.20` — the legacy strict gate) and `PORTFOLIO_CANDIDATE_THRESHOLDS` (`max_fold_negative_ic_streak=4`, `max_drawdown=-0.10` — looser streak in exchange for tighter DD). `THRESHOLDS_BY_ARM_CATEGORY` is the lookup the latest-stack worker uses to dispatch by `ArmSpec.category`. `AlphaEligibilityThresholds` gained a `name` field so the evidence JSON identifies which set was applied. v5 universe-300 rerun shows **G now passes eligibility** under the portfolio_candidate gate.
- **F-arm ablation** (long-only top-30 *without* learned-PCA) is the natural next experimental step and the data-driven `ARM_SPECS` dispatch makes it a one-tuple addition: `panel_key="pv_form"`, `requires_pca=False`, `portfolio_config_factory=_long_only_top30_config`. No new branch in the dispatch loop is required. Run it; if the long-only-without-PCA Sharpe ≥ Arm D's, learned-PCA stays research-status and the production candidate should be the F-arm shape.
- **Mining and paper-trade-replay opt-in to realized mode**: today only the latest-stack driver populates `realized_return_1d`. The mining driver (`scripts/mine_alphas.py`) builds samples with the same horizon shape; it should populate the realized return too, otherwise mining evidence stays in legacy mode and remains unsuitable for portfolio-style metrics. Likely a 1-day follow-up.

## Action Items

1. [x] Extract `has_realized_returns(scored)` helper, colocated with `SupervisedAlphaSample`, used by both evaluators. (Closes review finding #4.) — shipped in commit `02090032`.
2. [x] Add a regression test that loads the saved evidence JSON and asserts the required schema keys are present — pin the contract. (Closes review finding #5.) — shipped in commit `02090032` (`TestEvidenceJsonSchema`).
3. [x] Add a regression test that exercises cross-fold weight-carry in realized mode (two folds, second fold's rebalance turnover compares to first fold's held weights). (Closes review finding #6.) — shipped in commit `02090032` (`TestCrossFoldWeightCarry`).
4. [x] Make the PCA-fit-failure-skipped arms return a non-zero exit code or surface in a final WARNING summary. (Closes review finding #2.) — shipped in commit `02090032`; the script now tracks `skipped_specs` and returns exit code 3 if any were skipped.
5. [x] Replace the `if/elif` arm dispatch chain with a builder callable on `ArmSpec`. (Suggestion S1.) — shipped in commit `02090032`; `ArmSpec` now carries `panel_key` + `portfolio_config_factory` and the dispatch loop is fully data-driven.
6. [x] Emit `realized_mode_used: bool` on the evidence so dashboards can detect a quiet downgrade to legacy mode. (Suggestion S2.) — shipped in commit `02090032`. `cli_args` also added so the run is fully reproducible from the evidence file alone.
7. [ ] Open a follow-up ADR (ADR-004) for the trading-day fold generator once a concrete need surfaces. (Deferred — the sample-level purge in this PR fixes the leakage; trading-day fold boundaries are a follow-up that touches every walk-forward caller.)
8. [ ] Open a follow-up task to populate `realized_return_1d` in the mining driver so mining evidence can also gate on realistic Sharpe / drawdown. (Deferred — `scripts/mine_alphas.py` builds samples with the same horizon shape and should opt in next, otherwise mining evidence stays in legacy mode and remains unsuitable for portfolio-style metrics.)

## Related

- [ADR-001](adr-001-operational-hardening.md) — Operational hardening. Its "every signal class has IC ≈ 0" framing was an artefact of the v1 return-accounting bug this ADR closes; ADR-001 carries a status update pointing here.
- [ADR-002](adr-002-learned-family-representation-choice.md) — Learned-family representation choice (the v1 → v2 PCA standardisation fix that immediately preceded this one). The corrected backtest in this ADR is the first walk-forward evidence available against the v2 PCA artifact.
- Memory: [Backtest Latest Stack](../../memory/project_backtest_latest_stack.md) — v1 vs v2 result comparison.
- PR [#62](https://github.com/Liu-Ming-Yu/Quant/pull/62) merged 2026-05-27. Commits on branch `fix-return-accounting-and-purge`:
  - `618994b7` — core fix (sample model, evaluators, walk-forward, latest-stack driver).
  - `7b723536` — this ADR's first draft.
  - `02090032` — cleanup commit closing all 12 audit findings (action items 1–6 above).
  - `9ad4fabf` — CI cleanup (ruff + mypy gates).
- Tests: `tests/unit/research_service/campaigns/test_return_accounting_fix.py` — 36 regression tests pinning the separation, schema, validation, and registry invariants.
