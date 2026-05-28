# ADR-002 — Representation choice for the learned feature family

**Status:** Accepted (2026-05-26). v1 shipped in PR #58; the artifact schema bumped v1 → v2 in PR #61 (2026-05-26) when standardisation of source features was added — the decision (PCA over autoencoder/MoE/etc.) is unchanged, but the artifact gained a per-feature `scale` field.
**Context:** Phase 10 of the Quant Feature Expansion brief.

## Context

The brief's Phase 10 calls for a "learned representations" feature
family that consumes the 9 prior families' source features and emits
a smaller set of learned representations. The implementation has two
hard constraints from the platform's stability + governance contract:

1. **The family must NOT fit / train during `compute_*` calls.**
   Fitting belongs to the model / research trainer. The feature
   family is only a deterministic artifact-backed transform.
2. **The artifact must be versioned, persistable, and auditable.**
   A future auditor must be able to load the artifact and reproduce
   the same outputs months later.

This ADR records why **PCA** was chosen as the v1 representation,
and why four canonical alternatives — **autoencoder**,
**mixture-of-experts** (MoE), **XGBoost leaf indices**, and
**explicit interaction expansion** — are deferred.

## Decision

`learned-representations-v1` is an **artifact-backed PCA transform**.
The v2 schema bump (PR #61) adds input standardisation but does not
change the representation choice — every "Why deferred" argument below
still holds.

* The artifact (`PCAArtifact`) is a frozen dataclass carrying:
  - `n_components` (8)
  - `feature_names` (source columns in fit order)
  - `mean` (n_features,) — sample mean of the raw inputs
  - `scale` (n_features,) — sample std of the raw inputs **(v2 addition)**.
    The compute path standardises with `(x − mean) / scale` before
    projecting. Without this, sklearn's PCA found a single dominant
    direction driven by feature-scale variance (EVR PC1 ≈ 1.0); after
    standardisation the EVR is spread across 8 components.
  - `components` (n_components, n_features)
  - `explained_variance_ratio` (n_components,)
  - `fit_metadata` (str → str), including the artifact schema version.
* The trainer (`fit_pca_artifact`) lazy-imports `sklearn`, fits a PCA
  on a *standardised* training panel, bakes the raw-space `mean` /
  `scale` into the artifact, and emits a `PCAArtifact`. The trainer
  is invoked **out-of-band** by a research workflow — never by the
  compute path.
* The compute path (`compute_learned_features`) reads the artifact,
  validates compatibility, standardises with `(x − mean) / scale`,
  and does one matmul (project) + one matmul (reconstruct) + one L2
  norm per row in standardised space. No random initialization. No
  fitting. Same input × same artifact = same output, byte-for-byte.
* Output: 8 PC scores + 1 reconstruction error = **9 features**.

## Alternatives considered (and why deferred)

### Autoencoder

A small neural network with an encoder/decoder bottleneck. Captures
non-linearities that PCA cannot.

**Why deferred:**

- **Framework dependency.** Even a "frozen" autoencoder is a
  `torch.nn.Module` (or `tf.keras.Model`) with framework-specific
  state. Loading a frozen autoencoder requires installing
  `torch` (~1GB on Windows, ~500MB on Linux) as a runtime dependency
  of the feature compute path — versus PCA's runtime cost: zero new
  dependencies beyond numpy.
- **Determinism on free-floating loss landscapes.** Training is
  non-convex; the same training data + the same hyperparameters
  produce different weights on different seeds / hardware / library
  versions. PCA's SVD is deterministic up to a sign flip — much
  easier to pin in a CI artifact.
- **Audit story.** A 4-array PCA artifact is human-readable JSON
  (a few KB on 8×147). An autoencoder's weights are an opaque
  binary blob.
- **Premature for evidence-gated v1.** No walk-forward evidence yet
  exists that the platform's panel has structure PCA misses but an
  autoencoder captures. Until the v1 evidence base shows PCA leaves
  alpha on the table, the added complexity is unjustified.

**Promotion trigger:** ship autoencoder as `learned-representations-v2`
**only if** v1 walk-forward evidence shows PCA's reconstruction error
is informative as a forward-return predictor *and* a non-linear
representation would explain meaningfully more variance.

### Mixture of Experts (MoE)

A gating network selects among multiple expert sub-models per row;
captures regime-conditional structure.

**Why deferred:**

- **Two artifacts to govern.** Gate + experts both need
  trainer / artifact discipline; the PCA framing in this ADR covers
  exactly one frozen transform.
- **The regime detection is doing the heavy lifting.** Most of the
  value claimed for MoE in the alpha literature comes from the
  *gating* signal (regime classification) more than from per-regime
  representations. If regime is the goal, ship a regime-detector
  family (consuming `macro` features) first, then condition the
  learned transform on it in v2.
- **Compute-path complexity.** MoE's forward pass is
  `gate(x) @ stack([expert_i(x)])` — multiple matmuls,
  non-deterministic if the gate has stochastic routing. PCA is one
  matmul.
- **No clean PIT story.** MoE expert assignments depend on the
  current row's features, which leak across fold boundaries unless
  the gate is fold-aware. PCA's per-row transform is fold-independent
  by construction.

**Promotion trigger:** ship MoE as a `learned-representations-v3` **only
after** (a) a regime-detector family is in place, (b) v1/v2 evidence
shows regime-conditional behaviour, and (c) the gating mechanism is
made deterministic per-row.

### XGBoost leaf indices

Train a gradient-boosted forest on a forward-return target, then
emit per-row leaf indices as categorical features (or one-hot
encodings of leaf paths). Captures complex non-linear interactions
through tree paths.

**Why deferred:**

- **Supervised, not unsupervised.** PCA is unsupervised — the
  artifact is independent of any forward-return label. XGBoost leaf
  indices are supervised, which means:
  1. The family contract has to take a label column or pre-trained
     model as input — a major contract bump beyond v1's clean
     `(panel, artifact) → FeatureFrame` shape.
  2. Different label choices (next-day return, 5-day return,
     residual return) produce different leaf indices. The family
     proliferates artifacts per (label-horizon, target-type) pair.
  3. Labels make the artifact path-dependent on the target
     definition — much higher governance overhead.
- **Curse of dimensionality on leaf paths.** Even a modest 100-tree
  forest with depth-6 trees has 100 × 64 = 6,400 possible leaf
  positions per row. One-hot would blow the feature catalogue;
  index encoding makes the features categorical not numeric.
- **Runtime dependency.** Adding xgboost to the *compute path*
  (not just the trainer) raises the same dependency concern as
  the autoencoder.

**Promotion trigger:** the platform already has a separate
`formulaic-mining` track (Phase 4) that searches *over* features —
that's the right place for XGBoost-style learned interactions.
Revisit only if mining proves inadequate AND the operator wants
the leaf-index representation cataloged alongside other features.

### Explicit interaction expansion

Compute `feature_i × feature_j` for every pair (and possibly
`feature_i / feature_j`), then ship the expanded set as features.
Captures pairwise non-linearities with no training.

**Why deferred:**

- **Combinatorial explosion.** With 147 source features:
  - Pairwise products: 147 × 146 / 2 = 10,731 features.
  - Pairwise ratios: 147 × 146 = 21,462 features (asymmetric).
  - Triplets: ~half a million.
- **No supervision = no selection.** Without a target, there's no
  principled way to pick which pairs matter. PCA at least picks
  directions that maximize variance — a defensible unsupervised
  criterion. Pairwise expansion has no such ordering.
- **Storage + compute cost.** Each pairwise feature is a full panel
  column. 10K extra features × len(panel) doubles or triples the
  platform's feature-storage cost with no a-priori evidence the
  expansion contains alpha.

**Promotion trigger:** ship explicit pairwise expansion only when
the formulaic-mining track has identified specific high-IC pairwise
combinations to canonise. At that point each promoted pair lives in
the `formulaic` family with its own evidence, not in `learned`.

## Consequences

**Positive:**
- `learned-representations-v2` ships with a 5-array JSON artifact
  (`mean`, `scale`, `components`, `explained_variance_ratio` plus the
  `feature_names` tuple), a standardise-then-matmul compute path, and
  no new runtime dependencies.
- The audit story is trivial: cat the JSON, inspect the loadings.
- PIT safety is enforced by construction — the trainer is out-of-band,
  the compute path is deterministic.
- Reconstruction error is a free anomaly-detection signal alongside
  the 8 PC scores.

**Negative:**
- PCA cannot capture nonlinearities. If the platform's panel has
  structure PCA misses, v2 will leak signal. Evidence gating catches
  this — promoting v2 to a directional spec requires walk-forward IC
  that distinguishes "PCA captures the alpha" from "PCA destroys
  it." ADR-003's corrected backtest shows learned-PCA (Arm C) **does
  not** beat the formulaic-only baseline (Arm B) on Sharpe in the
  latest-stack run, so PCA stays research-status only until an
  ablation gate is built.
- PCA components are sign-invariant — an artifact retrained from
  identical data can flip every component's sign. Downstream models
  must be sign-robust (linear models with both signs in the feature
  set; tree models naturally robust).

## References

- Phase 10 brief (private — operator instruction)
- ADR-001 — Operational hardening (related stability discipline)
- [ADR-003](adr-003-return-accounting-separation.md) — Return-accounting
  separation. Its corrected backtest is the first walk-forward evidence
  available against the v2 PCA artifact; the ablation finding (Arm C
  loses to Arm B on Sharpe) frames whether PCA should be promoted.
- PR #58 (v1) and PR #61 (v1 → v2 standardisation).
- The other 9 feature-family docs under `docs/*-family.md`.
