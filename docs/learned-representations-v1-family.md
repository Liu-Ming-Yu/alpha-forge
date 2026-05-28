# `learned-representations-v1` Feature Family

> Definitive reference for the artifact-backed PCA representation
> family. The **first family in the catalogue that's
> representation-only** — it transforms features emitted by the other
> 9 families through a frozen `PCAArtifact`, producing 8 principal-
> component scores plus a reconstruction error. **No fitting happens
> inside this family at compute time** — the artifact is trained
> out-of-band by `fit_pca_artifact` and persisted as JSON.

## At a glance

| Field | Value |
|---|---|
| Family name | `learned` |
| Family version | `learned-representations-v1` |
| Source files | `src/quant_platform/research/features/learned/` |
| Public entry point | `compute_learned_features(panel, artifact, config)` |
| Required input | `PCAArtifact` (out-of-band-trained) + a panel carrying every name in `artifact.feature_names` |
| Feature count | **9** (8 PCs + 1 reconstruction error) |
| Tests | `tests/unit/research_service/features/learned/` (31 tests) |
| Compute-path dependencies | `numpy`, `pandas`. **No sklearn.** |
| Trainer dependencies | `scikit-learn` (lazy import in `trainer.py` only) |

## The 9 features

| Feature | Source |
|---|---|
| `learned_pc_1` | First PCA score |
| `learned_pc_2` | Second PCA score |
| ... | ... |
| `learned_pc_8` | Eighth PCA score |
| `learned_reconstruction_error` | L2 norm of `(x − x_hat)` per row |

All evidence-gated. See ADR-002 for why PCA was selected over
alternatives.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  TRAINER (out-of-band, research workflow)                             │
│                                                                       │
│   training_panel ──► fit_pca_artifact(...) ──► PCAArtifact            │
│   (sklearn PCA)         lazy-imports                                  │
│                         scikit-learn                                  │
│                                                                       │
│                                       │                               │
│                                       ▼                               │
│                            save_pca_artifact(path)                    │
└──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              artifact JSON on disk
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  COMPUTE PATH (every panel build)                                     │
│                                                                       │
│   load_pca_artifact(path) ──► PCAArtifact                             │
│                                                                       │
│   panel + artifact ──► compute_learned_features(panel, artifact)      │
│                            │                                          │
│                            ▼                                          │
│                       FeatureFrame (9 cols)                           │
│                                                                       │
│   No fitting. No sklearn. One matmul + one L2 norm per row.           │
└──────────────────────────────────────────────────────────────────────┘
```

## Input artifact contract

```python
@dataclass(frozen=True)
class PCAArtifact:
    artifact_version: str                      # "pca-artifact-v2"
    family_version: str                        # "learned-representations-v1"
    n_components: int                          # = 8 for v1
    feature_names: tuple[str, ...]             # source columns in fit order
    mean: tuple[float, ...]                    # (n_features,)
    scale: tuple[float, ...]                   # (n_features,) — per-feature std-dev
    components: tuple[tuple[float, ...], ...]  # (n_components, n_features)
    explained_variance_ratio: tuple[float, ...]
    fit_metadata: dict[str, str]               # provenance
```

All numeric fields are **tuples** (not numpy arrays) so the dataclass
is truly frozen and the JSON round-trip is lossless. The compute
path lifts tuples to ndarrays via `mean_as_numpy()` /
`scale_as_numpy()` / `components_as_numpy()`.

The `scale` field was added in `pca-artifact-v2` (2026-05-26) after
a full-stack backtest revealed that fitting PCA on a raw mixed-scale
panel collapsed the EVR onto PC1. The trainer now standardises by
default; the compute path divides `(x - mean)` by `scale` before
projecting. Operators who have already standardised out-of-band can
pass `standardise=False` to `fit_pca_artifact`, which writes a
`scale` of all-ones — the compute math then reduces to the v1
mean-centred projection.

**Validation (in `__post_init__`):**

- `artifact_version` must equal `ARTIFACT_SCHEMA_VERSION` (`"pca-artifact-v2"`).
- `family_version` must be non-empty.
- `n_components > 0`.
- `feature_names` non-empty + unique.
- `len(mean) == len(feature_names)`.
- `len(scale) == len(feature_names)` and every entry is strictly positive.
- `len(components) == n_components`; every row has length `len(feature_names)`.
- `len(explained_variance_ratio) == n_components`.

**Compute-time checks (additional):**

- `artifact.family_version == config.version` — refuses to apply a
  v2-trained artifact under v1.
- `artifact.n_components == config.expected_n_components` (= 8 for v1).
- `panel` must carry every name in `artifact.feature_names`.

## NaN handling

If any source feature is NaN on a row, **all 9 output features on that
row are NaN**. This is the conservative PIT-safe propagation rule:

- Don't impute (would introduce a hidden choice that affects the
  transform deterministic-ness).
- Don't drop (would lose row alignment with the input panel).
- Just propagate (let downstream models handle NaN explicitly).

Implementation: pre-compute a per-row NaN mask, substitute zero into
the input matrix to avoid NaN propagating through the matmul on
some BLAS implementations, then explicitly overwrite the masked
rows' output to NaN.

## Compute pipeline

```text
panel + artifact + config
        │
        ▼
_validate_artifact_compatibility (family_version match, n_components match)
        │
        ▼
_validate_panel_has_artifact_features (every artifact.feature_names in panel)
        │
        ▼
Extract source matrix in artifact's column order: X = panel[feature_names].to_numpy()
        │
        ▼
nan_mask = isnan(X).any(axis=1)  # per-row mask
        │
        ▼
X_safe = X with NaN rows zeroed
X_centered = X_safe − mean
pc_scores = X_centered @ components.T            # (n_rows, n_components)
X_recon = pc_scores @ components + mean
residual = X − X_recon                            # original X, NaN on bad rows
recon_err = L2 norm of residual per row (zero for NaN rows)
        │
        ▼
Mask NaN rows on outputs to NaN
Replace ±inf with NaN
        │
        ▼
FeatureFrame (instrument_id + date + 8 PCs + reconstruction_error)
```

## Operator quickstart

### One-time trainer step (research workflow)

```python
from pathlib import Path
from quant_platform.research.features.learned.trainer import fit_pca_artifact
from quant_platform.research.features.learned.loader import save_pca_artifact

# Build the training panel by concatenating the 9 prior families'
# FeatureFrame outputs (or load it from your research store).
training_panel = ...  # DataFrame with all source features

artifact = fit_pca_artifact(
    panel=training_panel,
    feature_names=("ret_1d", "ret_5d", ..., "real_yield_10y"),  # 147 names
    n_components=8,
    family_version="learned-representations-v1",
    extra_metadata={
        "training_window_start": "2018-01-01",
        "training_window_end": "2022-12-31",
        "n_instruments": "300",
    },
)
save_pca_artifact(artifact, Path("artifacts/learned/pca_v1.json"))
```

This requires `pip install scikit-learn` for the trainer's lazy
import.

### Compute step (every panel build)

```python
from quant_platform.research.features.learned import (
    DEFAULT_CONFIG,
    compute_learned_features,
    load_pca_artifact,
)

artifact = load_pca_artifact(Path("artifacts/learned/pca_v1.json"))

ff = compute_learned_features(
    panel=joined_feature_panel,  # carrying every artifact.feature_names column
    artifact=artifact,
    config=DEFAULT_CONFIG,
)

ff.frame              # 11-column DataFrame (instrument_id + date + 9 features)
ff.coverage           # per-feature notna() count
```

The compute path doesn't need `sklearn` — only `numpy` and `pandas`.

## Direction conventions and evidence gating

All 9 features ship `expected_direction="unknown"`,
`larger_is_better=False`. Two reasons:

1. **PC scores are sign-invariant.** PCA's underlying SVD pins the
   orientation of each axis up to a sign flip. A retrained artifact
   from identical data can produce components with the opposite sign,
   negating every PC score without changing the information content.
2. **Reconstruction error is genuinely directional** (low = expected;
   high = anomaly) but its **predictive direction** on forward returns
   is empirically uncertain — high reconstruction error could signal
   buying opportunities (overreaction) or selling opportunities
   (regime shift) depending on the regime.

Promotion of any feature to a directional spec is a family-version
bump (`learned-representations-v2`).

## What's deferred

See [ADR-002](architecture/adr-002-learned-family-representation-choice.md) for
the full rationale on why these were NOT chosen for v1:

- **Autoencoder** — non-linear representation. Deferred: framework
  dependency, non-deterministic training, opaque artifact.
- **Mixture of Experts** — regime-conditional representations.
  Deferred: needs a regime detector first; gating non-determinism.
- **XGBoost leaf indices** — supervised non-linear interactions.
  Deferred: belongs in the `formulaic-mining` track, not here.
- **Explicit interaction expansion** — combinatorial pairwise/triplet
  products. Deferred: 10K+ features per pairwise expansion with no
  selection criterion.

## Where to look next

- ADR: [`architecture/adr-002-learned-family-representation-choice.md`](architecture/adr-002-learned-family-representation-choice.md)
- Code: `src/quant_platform/research/features/learned/`
- Tests: `tests/unit/research_service/features/learned/` (31 tests)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
