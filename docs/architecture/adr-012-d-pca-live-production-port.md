# ADR-012 — Arm D (learned-PCA) live production port

**Status:** Accepted (in progress) — 2026-05-29
**Context:** [ADR-011](adr-011-live-pv-formulaic-feature-port.md) (pv+formulaic live port), [ADR-004](adr-004-per-category-eligibility-thresholds.md) (gate v3), `project_dollar_volume_scoring_defect`.

## Context

After the dollar-volume scoring fix (ADR-011) and the gate v3 redesign (ADR-004), the corrected walk-forward shows **no portable *linear* arm clears the Sharpe≥1.0 gate** — F/G sit at ~0.85, and adding orthogonal fundamentals (Arm P) lifts IC +20% but leaves Sharpe flat. The only arms that clear the gate are non-linear: **D (learned-PCA linear ranker, Sharpe 1.09)** and **N (GRU, 1.12)**. Operator decision (2026-05-29): build **D** as the first production non-linear port (N/GRU as a follow-up once D is stable in paper soak); parallelize IC→Sharpe construction research only after D's deployable path is unblocked.

D = `long_only_top30_pv_formulaic_learnedpca` (under v3: `learnedpca_streakdial` variants D/E). Construction: the pv+formulaic features (27 pv + 9 formulaic = 36) are projected through a **PCA artifact** (8 components + reconstruction error = 9 learned features) fit on the warmup window; the linear IC-weighted ranker (with the ADR-011 rank-normalized scoring) ranks on those 9 learned features; long-only top-30 + streak dial.

## Why this is tractable (de-risked 2026-05-29)

The PCA artifact is **pure data**: `PCAArtifact` is a frozen dataclass of `feature_names`, `mean`, `scale`, `components`, `explained_variance` (tuples of floats). Serialization exists (`loader.save_pca_artifact`/`load_pca_artifact`, JSON; schema `pca-artifact-v2`). The transform (`compute_learned_features`) is **deterministic matmul** (center → scale → project), no sklearn/pickle at inference. The fit is **static** (warmup-only, not per-fold rolling). Verified end-to-end: a persisted artifact round-trips exactly and reproduces the transform exactly (`np.allclose`). So "manifest/artifact loading" and "deterministic replay" are sound by construction — the inference path has no stochastic or unportable component.

## Requirements (operator-specified deliverables)

1. **Live-inference path** — live bars → pv+formulaic (ADR-011, done) → PCA transform → 9 learned features → score with D's frozen weights → top-30.
2. **Manifest/artifact loading** — a versioned, persisted production PCA artifact + manifest the live engine loads (not re-fit live).
3. **Deterministic replay** — same bars + same artifact + same weights ⇒ identical holdings.
4. **Paper-trading adapter** — D routes through the existing IB-paper execution path (ADR-011 increment-3 plugin framework + risk limits).
5. **Gate monitoring** — live IC / drawdown / streak tracked against the v3 gate so degradation surfaces.

## Architecture constraint (the layering crux, same as ADR-011)

The learned-PCA code lives in `quant_platform.research.features.learned` (composition layer). The live engine is in the inner layers (`services`/`engines`), which **must not import `quant_platform.research`** (`check_import_boundaries.py`). So the **inference** modules must be kernel-extracted to the inner layer; the **sklearn trainer stays in research** (offline fitting is a research concern). This mirrors ADR-011's "compute moves to the kernel, fit/register stays in research" split exactly.

- **Move to `services/research_service/features/kernel/learned/`:** `artifact.py` (PCAArtifact + schema), `loader.py` (JSON save/load), `features.py` (`compute_learned_features` — the transform), `config.py` (`LearnedConfig`). All depend only on kernel contracts/transforms + numpy/pandas/json — pure, extractable.
- **Stays in `research.features.learned`:** `trainer.py` (`fit_pca_artifact`, sklearn). Re-export shims at the old `research.features.learned.*` paths preserve every importer (the backtest, tests).

## Incremental sequence (each its own verified-green commit)

- **Increment 1 — artifact + manifest foundation (DONE 2026-05-29).** `scripts/build_live_pca_artifact.py` fits the production PCA artifact on all available history (the live warmup) over universe-300 pv+formulaic, and persists it + a manifest (artifact schema, source feature names + version, the 9 learned output names, n_components, fit window, bars fingerprint, git commit, timestamp) under **`infra/artifacts/learned_pca/`** (a *tracked* location — `data/` is gitignored, but the live engine must load a versioned artifact that travels with the repo for deterministic deploy). The builder reuses the backtest's exact pv+formulaic compute (no divergent copy) and self-checks the round-trip before writing. Re-run quarterly to mint a new version. Produced: 8 components, 36 source features → 9 learned features, fit window 2021-12-27 → 2026-05-22, 330 instruments. Proven: persisted artifact round-trips + reproduces the transform exactly.
- **Increment 2 — kernel-extract the learned inference modules (DONE 2026-05-29).** `git mv` of `artifact.py` (pure, verbatim), `config.py`, `loader.py`, `features.py` → `services/research_service/features/kernel/learned/`, with their `research.features.{contracts,transforms,learned.*}` imports rewritten to `kernel.*`. Re-export shims left at the four `research.features.learned.*` paths (object-identical: `shim.compute_learned_features is kernel.compute_learned_features`). The `__init__` (`register_family(MANIFEST)`) and `trainer.py` (sklearn `fit_pca_artifact`) stay in research and import through the shims. Verified: `mypy src` clean (978), import-boundaries clean, ruff clean; family still registers (`learned/learned-representations-v1`), trainer imports, both scripts import, 47 learned + 733 research-feature tests green. The live learned family (increment 3) can now import the transform from the inner kernel without crossing `services → research`.
- **Increment 3 — live D feature family + bundle.** A `learned_pv_formulaic` family: live bars → pv+formulaic (ADR-011 kernel) → load the manifest artifact → `compute_learned_features` → `FeatureBundle` of the 9 learned features (latest row per instrument). Golden-master parity vs the backtest's D features on identical bars.
- **Increment 4 — D strategy plugin + wiring.** `arm_d` `BuiltInStrategyPlugin` (`feature_set_version` = the learned family version, `required_features` = the 9 learned names, `default_factor_weights` = D's frozen promoted weights, top_n=30); `--engine` choice; session risk limits (reuse ADR-011's `QP__RISK__MAX_GROSS_EXPOSURE=0.22`).
- **Increment 5 — simulated-backend validation + gate monitoring.** Reconcile live D top-30 + weights vs the D backtest (the increment-4-style parity check that caught the dollar-volume defect); wire live gate monitoring. On a clean reconcile → ib-paper soak.
- **Promotion.** Re-run the gate on D's corrected evidence, promote D to the registry (supersede — the registry currently has no active G after the 2026-05-29 demotion), with the artifact manifest pinned in the model metadata.

## Trade-offs / revisit

- The artifact is fit on a fixed warmup. Live, it should be **periodically re-fit** (e.g. quarterly) as new history accrues — a scheduled offline job that produces a new manifest version; the live engine pins a version for deterministic replay. Increment 1's builder is that job's core.
- D carries a reconstruction-error feature; confirm it is in D's promoted weight vector (it is a learned feature, weight ≥ 0 under non-negative IC).
- If the kernel extraction surfaces a research-only entanglement in `features.py` (e.g. a registry side-effect like the pv/formulaic `__init__`), apply the same registration-split as ADR-011 (compute moves, `register_family` stays).

## Related
- [ADR-011](adr-011-live-pv-formulaic-feature-port.md) — the pv+formulaic port this extends (its kernel + plugin framework are reused).
- [ADR-004](adr-004-per-category-eligibility-thresholds.md) — the v3 gate D is evaluated against.
- `scripts/build_live_pca_artifact.py` — the production-artifact builder (increment 1).
