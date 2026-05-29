# ADR-011 — Porting the pv+formulaic feature families to the live engine (G-live integration)

**Status:** Proposed (2026-05-28). Opens the workstream to make the live paper engine trade Arm G's construction so the 30/90/365-day paper soak measures the promoted model rather than a different strategy.

## Context

Arm G (`long_only_top30_pv_formulaic_streakdial`) is the production-lead research arm and is now the active model in the registry (ADR-004 + the registry-promotion adapter). The next milestone is the paper soak. But verifying the live wiring surfaced a gap:

- The live `supervise` path runs the **`cross_sectional_equity`** engine under a generic `cli_cycle` strategy run, building its signal from **engine-config `factor_weights`** over the **`close`** feature family (`momentum_1m`, `vol_compression`, …). It does **not** execute G's construction.
- G's signal is a linear combination of **20 price_volume + formulaic features** (`close_to_open_return`, `high_low_range_20d`, `dollar_volume_20d`, `mom_12_1`, `mom_3_1`, …) with G's promoted IC-weights.
- The live feature pipeline (`services/research_service/features/plugins`) registers only the **`close`, `catalyst`, `event`, `composite`** families. **There is no `price_volume` or `formulaic` family computed live.**

So a faithful "G-live" strategy plugin is blocked: declaring G's 20 features as `required_features` would fail the engine's `schema_guard` every cycle (fail-closed → no trading), because nothing produces them.

## The architectural barrier

The obvious fix — have the live feature pipeline call the existing research feature math — is **forbidden by the import-boundary checker**:

- `scripts/check_import_boundaries.py`: `quant_platform.research` is a **composition layer**; `services` is an **inner layer**; an inner layer must not import a composition layer (fails hard).
- `quant_platform.research.features.price_volume` / `…formulaic` (the pv+formulaic math) live in the **research composition layer**.
- `services/research_service/features/plugins` (the live feature-family registry) is in the **services inner layer** and imports nothing from `quant_platform.research` today.

Therefore the live pipeline cannot reuse the research math in place. The feature computation must be **ported into an inner layer** the live pipeline may import.

## Decision (proposed)

Build the G-live integration in tested increments. **Do not point `ib-paper` at it until the simulated-backend validation in step 4 passes.**

1. **Port the pv+formulaic feature computation into `services/research_service/features`** (inner layer) as a new live feature family `pv_formulaic`. Reuse the *algorithms* from `quant_platform.research.features.{price_volume,formulaic}` but relocate/duplicate the math so the services layer owns it (no cross-layer import). Expose a `BundleFeatureComputer` whose `builder(payloads, as_of) -> FeatureBundle` converts the live `MarketBar` payload (`BARS_EOD_INPUT`) → the OHLCV frame the math expects → per-instrument feature vectors for the latest `as_of` row. Register it in `build_research_feature_family_plugins`.
   - **Crux — transform parity.** The live feature values must match the transform G was fit on (raw vs cross-sectionally rank/z-scored vs sector-neutralised; see `features/neutralization.py`). If the live family rank-normalises but G's research pipeline z-scored, the live engine trades a *different* signal than G. This parity must be asserted by tests against the research computation on shared fixtures — it is the highest-risk part.
2. **G strategy plugin.** Add a `BuiltInStrategyPlugin` (`engines/framework/plugins.py`): `required_features` = G's 20 names, `default_factor_weights` = G's frozen promoted IC-weights (freezing is the *correct* soak semantic — soak what was promoted), `default_max_positions=30`, 21-day rebalance. Register in `_PLUGINS` and add the key to the `supervise --engine` choices.
3. **Risk limits.** G's caps (`max_single_name_weight=0.05`, `max_gross_exposure=0.22`, `min_cash_buffer=0.05`) are applied via session `RiskLimits` (settings), not the constructor (`LongOnlyPortfolioConstructor` exposes only `top_n`). Configure them for the G run.
4. **Validate, then soak.** Run the plugin under `--execution-backend simulated` and reconcile its scores/holdings against G's backtest on a shared window before any `ib-paper` run.

## Known fidelity gaps (document, don't silently approximate)

- **Frozen weights vs per-fold refit.** Research G refits IC-weights each walk-forward fold; live uses the frozen promoted weights. Correct for a soak (you soak a fixed model); re-promote to refresh.
- **No streak dial live.** The `FoldStreakRiskConfig` exposure throttle is a research-construction feature with no live `LongOnlyPortfolioConstructor` equivalent. The live G is G-without-the-dial unless the dial is also ported.
- **Feature pipeline differences.** Even after the port, live bars (vendor, adjustment, as-of timing) may differ subtly from the research parquet snapshot.

## Options considered

- **A — Reuse research math via a cross-layer import (rejected):** violates the import-boundary checker (`services` ✗→ `research`); would require an approved exception and inverts the layering.
- **B — Port the math into `services` (chosen):** respects layering, makes the feature family a first-class live capability, and is testable for parity against the research source.
- **C — Config the existing `cross_sectional_equity` engine with G's weights (rejected):** its `close` family computes different features under different names; setting weights keyed to G's 20 features would fail `schema_guard`. No faithful path without the port.
- **D — Soak the generic `cross_sectional_equity` strategy instead (deferred):** a real *operational* soak (broker/NAV/reconciliation) but the traded signal isn't G's; doesn't satisfy "soak G."

## Consequences

- A meaningful, layering-clean port that gives the live engine a price_volume+formulaic feature family — reusable beyond G.
- The transform-parity tests become the contract that keeps live and research signals aligned.
- Until this lands, G's promotion is a **governance record**; the live engine does not trade G, and no paper soak measures G.

## System design (2026-05-28, `/system-design`)

### Requirements

**Functional**
- Live engine computes G's 20 features on live `MarketBar` payloads each cycle, as-of correct: **17 price_volume** (`close_to_open_return`, `high_low_range_1d/20d`, `dollar_volume_20d`, `volume_z_20d`, `drawdown_from_252d_high`, `distance_to_52w_high`, `overnight_gap`, `reversal_1d/5d`, `ret_21d/63d/126d/252d`, `mom_3_1/6_1/12_1`) + **3 formulaic** (`wq_alpha_002_paraphrase`, `wq_alpha_012`, `wq_alpha_041`).
- Output a `FeatureBundle` the `LinearWeightSignalModel` + `LongOnlyPortfolioConstructor` consume.

**Non-functional**
- **Layering:** `services` must not import `quant_platform.research` (composition).
- **No duplication** of feature math between research and live.
- **Transform parity:** live values must equal what G was fit on — the correctness crux.
- Per-cycle latency is modest (≈330 daily instruments); pandas compute is fine.

**Constraints (from code inspection)**
- pv+formulaic depend on shared `research.features.{contracts,registry,transforms}` (used by *every* family) and need the formulaic AST engine (`ast`/`operators`/`evaluator`/`panel`/`library`) for the 3 alphas.
- In-`src` importers of pv/formulaic are confined to `research/features/*`; the only external importer is `scripts/backtest_latest_stack.py`. Bounded blast radius.

### Architecture options

1. **Duplicate G's math in `services`.** Rejected — violates the no-duplication constraint and guarantees parity drift (two copies diverge).
2. **Move the whole feature factory to `services`.** Rejected — drags every family (fundamentals/learned/regime/text) + the registry; unbounded blast radius.
3. **Extract a pure feature-compute *kernel* to the inner layer, consumed by both research and live (chosen).**
   - New inner package (e.g. `services/research_service/features/kernel/`) holding the **pure math**: the `FeatureFrame`/`FeatureSpec` contracts, `transforms`, the price_volume compute, and the formulaic engine + curated `library`. Depends only on `core` + numpy/pandas.
   - `research.features.{contracts,transforms,price_volume,formulaic}` become **thin re-export shims** → kernel (composition→inner import, allowed). Mining, the other families, and the backtest keep working unchanged; the research registry/neutralization stay in research importing the shimmed contracts.
   - The live `pv_formulaic` family (`services`) imports the kernel directly.
   - **Parity by construction:** both layers execute the *same* kernel code, so the live signal matches G's research signal by construction. The only adaptation is bars→frame + as-of slicing — pinned by golden-master tests.

### Deep dive
- **Live family builder:** `build_pv_formulaic_feature_bundle(payloads, as_of)` → `MarketBar` payload to OHLCV frame (rows ≤ as_of) → kernel pv compute + kernel formulaic eval → latest row per instrument → `FeatureBundle.alpha_features` keyed by feature name. Registered in `build_research_feature_family_plugins` with a pinned `feature_set_version`.
- **G plugin:** `BuiltInStrategyPlugin` — `required_features` = G's 20 names, `default_factor_weights` = G's frozen promoted IC-weights, `top_n=30`, 21-day rebalance; registered in `_PLUGINS` + the `supervise --engine` choices.
- **Risk limits:** G's caps via session `RiskLimits` (single-name 0.05 / gross 0.22 / cash 0.05), not the constructor.
- **Parity harness (crux):** golden-master — run the kernel over the *same* bars the realized_v2 backtest used and assert per-instrument feature values equal the research feature-factory output within tolerance, and that the live family's latest-row values equal the research panel's last row for those (instrument, date) pairs.

### Sequencing (each its own PR)
1. **Kernel extraction + research shims** — behavior-preserving; the full research/mining/backtest suite must stay green. (The big refactor; verify `contracts`/`transforms` are pure first.)
2. **Live `pv_formulaic` family + parity golden-master tests.**
3. **G strategy plugin + `--engine` wiring + risk-limit config.**
4. **Simulated-backend validation** (reconcile live scores/holdings vs the G backtest) — *only then* `ib-paper`.

### Progress + refined pattern (2026-05-28)

Step 1 is landing as one verified, green-committed `git mv`+shim per module:
- **Done:** `contracts` (FeatureSpec/FeatureFrame/FamilyManifest + Literal aliases; 77 importers) and `transforms` (rolling/group helpers + tokens; 59 importers) moved to `kernel/`; both are pure (`core`/stdlib/numpy/pandas only), so `git mv` was verbatim. `mypy src` clean, import-boundaries clean, full research-ecosystem suite green at each step.
- **Refinement for `price_volume` / `formulaic` (the compute moves):** their package `__init__.py` calls `register_family(MANIFEST)` as an import side-effect — coupling them to the research `registry` (composition). So they do **not** move whole. Split each: the pure **compute** (`config.py`, `features.py`, and for formulaic the `ast`/`operators`/`evaluator`/`panel`/`library`/`serialization`) moves to `kernel/`, with its `research.features.{contracts,transforms}` imports rewritten to `kernel.*`; the **family-registration `__init__`** stays in `research.features.*` (a research concern), importing the compute from the kernel (`research → services`, legal). Per-submodule re-export shims preserve the existing importers. The formulaic **mining** subpackage stays in research and keeps importing the (shimmed) formulaic core.
- **Done (cont.):** `price_volume` compute move — `config.py`+`features.py` → `kernel/price_volume/` (imports rewritten to `kernel.*`), re-export shims left, the `register_family` `__init__` kept in research; verified green (mypy src 955, 1193 tests, compute identical via both paths, family still registers).
- **Done (cont.):** `formulaic` engine core — the 6 modules (`ast`, `operators`, `evaluator`, `panel`, `library`, `config`) → `kernel/formulaic/` with intra-package + contracts/transforms imports rewritten to `kernel.*` and per-module re-export shims; `features.py`/`auto_library.py`/`promotion.py`/`__init__` (register_family) and the `mining/` subpackage stay in research, importing the moved core via shims. Verified: family registers, `LIBRARY` carries G's wq alphas, evaluator identical via shim/kernel, mining imports; mypy src clean (962), 1265 tests green.
- **✅ Step 1 (kernel extraction) COMPLETE** — contracts + transforms + price_volume + formulaic all in the inner-layer kernel; the live pipeline can now import the feature math without crossing `services → research`, and parity is by construction (research consumes the same kernel via shims).
- **Increment 2 — compute + parity DONE.** `pv_formulaic/compute.py::compute_pv_formulaic_frame` reproduces the research pv+formulaic matrix on live bars via the kernel; the golden-master test asserts the `MarketBar→adapter→compute` path equals the research compute on identical data (`assert_frame_equal`) and produces the full 27 pv + 9 formulaic surface (incl. G's wq alphas). Parity-by-construction proven.
- **Increment 2b — DONE.** `pv_formulaic/family.py::build_pv_formulaic_feature_bundle` assembles a `FeatureBundle` (latest row per instrument) and is registered as the `pv_formulaic` family (`pv-formulaic-live-v1`) in `build_research_feature_family_plugins` — so the engine can now compute G's 27 pv + 9 formulaic features live. **Scoring-transform reconciliation settled: the bundle carries RAW features** (not rank-normalised like the `close` family). Evidence: `build_supervised_samples` feeds the research ranker raw values and `score_features` is a raw weighted sum `Σ feature·ic_weight`; so a raw bundle + G's promoted IC weights, scored by `LinearWeightSignalModel` (which rank-normalises the *score*), reproduces G's cross-sectional ranking — hence the top-N selection — by construction. Verified: golden-master compute parity, bundle carries raw latest-row values, registry has 5 families, mypy src clean (964), 1256 tests green.
- **Increment 3 — G strategy plugin + family wiring DONE.** Added the `arm_g` `BuiltInStrategyPlugin` (`name=arm_g_pv_formulaic_v1`, `feature_spec.version=pv-formulaic-live-v1`, `required_features`=G's 20 weighted names, `default_factor_weights`=G's frozen promoted IC weights from `realized_v2` summing to 1.0, `default_max_positions=30`, monthly rebalance) and wired it into `supervise`/`run-engine` `--engine` choices.
  - **Critical wiring fix (the family-selection gap):** the engine previously scheduled its feature job under a **hardcoded** `FEATURE_SET_VERSION` (the `close` family's `1.1.0`) at `engine_runner/__init__.py`, so the plugin's `feature_spec` was decorative and *every* plugin computed `close`. Added `feature_set_version` to `EngineConfig` + `BuiltInStrategyPlugin` (default `""` ⇒ engine falls back to `FEATURE_SET_VERSION`, preserving the close-family plugins unchanged) and threaded it through `create_runner`; engine_runner now registers under `config.feature_set_version or FEATURE_SET_VERSION`. So `arm_g` (which sets `pv-formulaic-live-v1`) is the first plugin whose feature job resolves to a non-`close` family (`family_for_version("pv-formulaic-live-v1") == "pv_formulaic"`). Backward compat asserted in tests (close plugins keep `feature_set_version == ""`).
  - **Risk caps are session settings, not plugin code.** `RiskLimits` is built once at session bootstrap from `PlatformSettings.risk` (`config_risk_execution/risk.py`). G's **single-name 0.05** and **cash-buffer 0.05** already match the defaults; only the **gross cap** differs (default 0.60). So the soak's only override is `QP__RISK__MAX_GROSS_EXPOSURE=0.22` at launch — which also matches the gross the backtest used (`targets.py`: `investable = min(max_gross_exposure, 1-min_cash_buffer)` ⇒ 0.22). `max_daily_turnover` 0.20 and `max_drawdown_halt` −0.15 comfortably bound G's 0.48% turnover / −4.21% DD. (A per-plugin risk-limit override would be a future enhancement if multiple strategies run concurrently with different caps.)
  - Verified: `mypy src` clean (964), ruff format+check clean, import-boundaries clean, 102 engine-wiring/CLI/plugin/feature tests green (incl. backward-compat + the new `arm_g` plugin/runner tests).
- **Increment 4 — simulated-backend validation FOUND A BLOCKING DEFECT. Live port PAUSED.** The parity check (real universe-300 bars, last available date) shows the live and backtest portfolios share only **3/30** names. Root cause is a latent **research-stack scoring bug**, not a port bug:
  - `score_features` (`campaigns/metrics/ranker_metrics.py`) is a **raw weighted sum** `Σ feature·weight` with **no cross-sectional normalization**, fed **raw** feature values by `build_supervised_samples`.
  - `dollar_volume_20d` is **raw dollars** (`close·volume`, median ≈ 1.9e8). Its score contribution (`median|val|·weight ≈ 1.08e7`) dwarfs the next feature (`volume_z_20d ≈ 0.087`) by ~**10⁸:1**. So the IC weights — fit by *rank-based* Spearman IC (scale-invariant) — are applied to *raw* values whose scales differ by 8 orders of magnitude.
  - **Measured consequence:** `Spearman(full-20-feature score, dollar_volume_20d alone) = 1.0000` (mean and min over 60 days); top-30(full) vs top-30(dollar_volume only) overlap = **30/30 every day**. **G's selection is identical to a pure `dollar_volume_20d` (liquidity/size) sort; the other 19 features are numerically inert.** G's reported evidence (oos_ic +0.243, Sharpe 1.0886) is therefore the performance of a dollar-volume sort on universe-300, not the documented 20-factor alpha. The IC (`_spearman(day_scores, labels)`, line 138) and the long-only weights (`score/denom_rebal`, line 175) both ride on this dollar-volume-dominated score.
  - **Live consequence:** `LinearWeightSignalModel.score` clamps `raw` to [-1,1] (`max(-1,min(1,raw))`); every raw score is ≫1 (range 1.3e6–1.5e9) → all 330 clamp to +1.0 → ties → arbitrary top-30 (the 3/30). So even the "replicate the backtest" goal is moot — the backtest itself is a dollar-volume artifact.
  - Reproduce: the diagnostic in this session (load universe-300 2024–2025 bars → `compute_pv_formulaic_frame` → score both ways). The compute/feature parity from increments 1–2 is unaffected and still correct; the defect is purely in *scoring on raw scales*.
  - **This is a fork for the operator (see "Decision required" below) — do NOT proceed to `ib-paper`.** The scoring must be fixed (cross-sectional normalization per date before the weighted sum) in the shared kernel, after which the backtest must be re-run, G re-validated/re-promoted (G may no longer be the lead — or may be revealed as a liquidity artifact with little real alpha), and the live weights re-derived. The narrowest first step is to **quantify how much real alpha survives once features are properly scaled** before committing to a re-run.

### Decision (2026-05-28): Option A chosen — fix scoring + full re-run + re-promote
Operator chose **(A)**. Implementation + results:
- **A1 (done):** `cross_sectional_rank_normalize` added to the scoring kernel (`features/kernel/transforms.py`) — per-date cross-sectional percentile rank in `[0,1]`, NaN→0.5 (median). `[0,1]` (not centered) so that with non-negative weights summing to 1 the score lands in `[0,1]` → the live clamp is a no-op and long-only `score/Σ|score|` weighting stays all-positive.
- **A2 (done):** applied inside `_FittedLinearRanker.score` (`campaigns/models/linear.py`) — group the scoring batch by as-of, rank-normalize each weighted feature within its cross-section, then weighted sum. **Surgical:** only the linear arms change; GBDT (I/J) and GRU (N) use their own model classes and untouched `samples.features`, so their evidence is **bit-identical** (trees/NN are scale-robust anyway). The fit (`fit_correlation_weights`, rank-based Spearman IC) is invariant to the monotonic transform, so fitted weights are unchanged — only scoring is corrected. Verified on real bars: feature contributions now ∝ weights, score range [0.17, 0.94], `Spearman(score, dollar_volume)` 1.0000→0.24.
- **A2 smoke (G-only, normalized) — G is dethroned but real:** oos_ic +0.243→**+0.159**, ic_60d +0.139→**+0.063** (>0.03 ✓), bootstrap_ic_p05 **+0.015** (>0 → IC significantly positive), Sharpe 1.089→**0.852** (✗ <1.0), fold_negative_ic_streak 4→**7** (✗ >4), decile spread 0.016→**0.0045**. `eligibility.passed = False`. **So G's "passes all 5 gates" verdict was largely the dollar-volume artifact; corrected, G has genuine but sub-threshold alpha.** Prediction: the new eligible lead is a scale-robust **model** arm (J GBDT-rank was eligible at Sharpe 1.05, unchanged by the fix; N GRU 1.12 but failed streak) — confirmed/refined by the full A–N re-run (A3, in progress, output `backtest_latest_stack_normfixed/`).
- **A3 (done) — full A–N normalized re-run (`backtest_latest_stack_normfixed/`, 63 folds, 907 daily obs): NO arm passes the v1 eligibility gates. The binding gate is `fold_negative_ic_streak ≤ 4` for 13 of 14 arms.** Full table (sharpe / streak / ic60 / oos_ic / failed-gate):
  - A 0.94 / 5 / 0.064 / 0.162 — streak + sharpe
  - B 0.93 / 7 / 0.063 / 0.159 — streak + sharpe
  - C 0.93 / 7 / 0.057 / 0.162 — streak + sharpe
  - **D 1.09 / 7 / 0.057 / 0.162 — streak only** (PCA)
  - E 0.95 / 7 — streak + sharpe
  - F 0.94 / 7 — streak + sharpe
  - G 0.85 / 7 / 0.063 / 0.159 — streak + sharpe (former lead, dethroned)
  - H 0.87 / 5 — streak + sharpe
  - **I 1.04 / 4 / 0.0278 / 0.119 — ic_60d only** (GBDT-MSE; the ONLY arm passing streak, but ic_60d 0.0278 < 0.03)
  - **J 1.28 / 9 / 0.053 / 0.268 — streak only** (GBDT-rank; best Sharpe + best oos_ic, worst streak)
  - K 0.79 / 7 — streak + sharpe
  - L 0.86 / 7 — streak + sharpe
  - M 0.89 / 7 — streak + sharpe
  - **N 1.12 / 5 / 0.059 / 0.172 — streak only** (GRU; Sharpe ✓, ic_60d ✓, streak 5 just over 4)
  - **Headline:** removing the dollar-volume artifact raised the negative-IC streak from 4 → 5–9 across the board. `dollar_volume_20d` is a *stable* sort (megacaps stay megacaps) so it rarely had long negative-IC runs — **the artifact was simultaneously inflating Sharpe AND suppressing the streak gate.** The real multi-factor alpha is noisier. So the streak gate (ADR-004), long the binding constraint, now rejects every arm. There is real alpha (oos-IC ~0.16; J 0.27) but no arm clears the v1 gates → **no clean promotable lead exists; do not auto-promote a gate-failing arm.**
  - **The v2 DD-conditioned streak gate (PR #83, this session — not yet on this branch) is the designed remedy:** it admits streak ≤ 6 when the drawdown *during the worst streak* is contained (≥ −2%). Only arms with streak ≤ 6 can benefit: A(5), H(5), I(4), N(5). Of those, **N (GRU) is the sole arm that could pass v2** (streak 5 ≤ 6, Sharpe 1.12 ✓, ic_60d 0.059 ✓) — *iff* its DD-during-worst-streak is contained. So promotability reduces to one crisp question: **is N's drawdown during its worst negative-IC streak shallow (≥ −2%)?** Answering it needs PR #83's `streak_containment` metric + v2 gate merged onto this branch. (I fails ic_60d; J/D have streak > 6.)
  - **A4 is BLOCKED on an operator decision (see "Decision required #2").**

**Broader implication:** this very likely explains the long-standing meta-finding (in `project_backtest_latest_stack`) that *only ranking-changing models (no-PCA / GBDT / GRU) move IC* — the linear ranker was pinned to `dollar_volume_20d` the whole time, so dial/regime/cost/weighting interventions couldn't move the IC. The corrected linear arms should now respond to feature/weight changes.

### Streak investigation (2026-05-28): episodic + contained, NOT a chronic ceiling
Per-fold IC analysis of the normalized re-run (`backtest_latest_stack_normfixed/`) settles whether the 5–9 streaks are a fixable regime issue or an alpha ceiling:
- **The streaks are episodic and regime-clustered, not chronic noise.** Shared negative-IC episodes (≥10/14 arms negative) line up with identifiable macro stress: 2023-03/04 (regional-bank crisis), 2023-09/10, **2024-07..11 (summer rotation / pre-election)**, **2025-04..06 (tariff selloff)**, 2026-01/02. The linear arms' worst 7-streak is the *same window for all of them* (2024-07-08..11-11); J's worst 9-streak is 2022-11..2023-04; N's worst 5-streak is 2023-01..04.
- **The drawdown DURING every arm's worst streak is tiny — 0.00% to −1.26%, all ≥ −2%.** The negative-IC streaks are *ranking-degradation* episodes with near-zero capital impact (the long-only book stayed positive/flat through them). This is exactly the case the **v2 DD-conditioned streak gate** (PR #83: streak ≤ 6 admitted when streak-DD ≥ −2%) was built to admit.
- **Models break in different regimes.** Linear arms crater in 2025-spring (G's fold IC hits −0.51) while **J/N stay positive there (J +0.16/+0.26, N +0.10)** — the GBDT/GRU models are robust to the episode that kills the linear ranker, and vice-versa (J/N worst in early-2023).
- **Applying v2 to the corrected evidence:** arms with streak ≤ 6 AND contained streak-DD = A, H, I, N. Of these, only **N (GRU) clears every other gate** (Sharpe 1.119 ✓, ic_60d 0.059 ✓, oos_ic 0.17 ✓, max_dd −4.95% ✓, streak 5 ≤ 6 ✓, streak-DD −0.20% ✓). A/H fail Sharpe (<1.0); I fails ic_60d (0.0278<0.03). So **N is the sole v2-gate-passing lead.**
- **Two important nuances for the port:**
  1. **N is a GRU sequence model.** The live port (kernel + `pv_formulaic` family + `arm_g` plugin + `LinearWeightSignalModel`) was built for a *linear* model. Promoting N requires a **GRU live-inference path** in the engine — a substantial new workstream, not the existing linear port.
  2. **The best *linear* (portable) arm D (PCA, Sharpe 1.09, streak-DD −0.30%) misses v2 by one fold** (streak 7 vs cap 6). The v2 cap of 6 was calibrated (PR #83) on the *old artifact* evidence; on the corrected evidence the 2024-summer episode is exactly 7 folds for the linear arms, so the cap is now the single binding lever between "nothing portable passes" and "D passes." Recalibrating the cap on corrected evidence is defensible and would admit a portable linear lead.

### Streak-cap recalibration (2026-05-28): the cap does NOT honestly land at 7
Per the operator's directive, re-ran the held-out calibration (`scripts/calibrate_eligibility_thresholds.py`, grid extended to streak 3–9) on the corrected evidence (`backtest_latest_stack_normfixed/`, calibration = first 252 OOS trading days ≈ 16 folds, validation = remaining 47):
- **The streak metric is regime-unstable.** Linear arms (D/F/G/K/L/M): calibration streak **3**, validation streak **7** — the 2024-summer 7-streak episode lives *entirely in the validation window*, so it is unforecastable from calibration data. J: cal 9 / val 5. N: cal 5 / val 4.
- **The only *stable* cap (cal admit-set == val admit-set) is 9** (`streak ≤ 9, dd ≥ −0.05` admits all 11 candidates on both windows). Every cap below 9 — **including 7** — is unstable. So a cap of 7 is NOT validated; setting it would be the exact overfit the hold-out is designed to catch (from the calibration window you'd have picked cap 4–5, which validation then violates).
- **Conclusion:** the streak gate does not survive honest recalibration on corrected evidence — at a meaningful cap it's unstable, and the stable cap (9 ≈ no gate) gives no discrimination. **Per the operator's conditional ("promote D iff the cap lands at 7"), the condition fails → D is NOT promoted.** With J excluded (no override) and N's GRU port deferred, the logical outcome is to **park the live port** pending genuine alpha/gate improvement, or to consciously redesign the gate. Reframes the streak from "binding-but-passable" to "not a robust out-of-sample discriminator on this universe/label."

### Decision required #2 (2026-05-28): no arm passes v1; streak is contained/episodic
Given the investigation (contained-DD regime episodes; N passes v2; D the portable linear arm misses the cap by 1 fold):
- **(1) Promote N (GRU) under the v2 gate** — the principled gate-passing lead. Cost: a GRU live-inference workstream before the soak.
- **(2) Recalibrate the v2 streak cap on corrected evidence** (held-out calibration, PR #83's script) — if the cap lands at 7, the *portable linear* arm D (Sharpe 1.09) passes and the existing port works with minimal change.
- **(3) Promote best arm for paper despite v1** (J Sharpe 1.28, or N) — explicit paper-only gate override to observe live behavior.
- **(4) Park the live port** — treat as alpha-improvement work; resume once a gate-passing portable arm exists.

### Trade-offs / revisit
- The kernel extraction is a sizable behavior-preserving move; the payoff is parity-by-construction + zero duplication. If `contracts`/`transforms` turn out entangled with research-only code, fall back to Option 1 for *just G's 20-feature compute*, accepting a heavier parity-test burden.
- The streak dial is not ported (documented gap above); revisit if the soak shows it matters.

## Related

- [ADR-004](adr-004-per-category-eligibility-thresholds.md) — the streak-gate settlement + registry promotion that made G the active model.
- `scripts/promote_latest_stack_arm.py` — the one-call promotion adapter (the upstream of this workstream).
- `scripts/check_import_boundaries.py` — the layering rule that forces the port (Option B over A).
