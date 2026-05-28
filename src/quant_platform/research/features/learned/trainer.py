"""Out-of-band trainer for the PCA artifact.

This module is **NOT** part of the feature-compute path. The
``learned-representations-v1`` family is intentionally
representation-only — :func:`compute_learned_features` consumes a
frozen :class:`PCAArtifact` and applies a deterministic transform.
Training / fitting happens here, ahead of time, in a research
workflow:

    from quant_platform.research.features.learned.trainer import fit_pca_artifact
    from quant_platform.research.features.learned.loader import save_pca_artifact

    artifact = fit_pca_artifact(
        panel=training_feature_panel,
        feature_names=SOURCE_FEATURE_LIST,
        n_components=8,
        family_version="learned-representations-v1",
    )
    save_pca_artifact(artifact, Path("artifacts/learned/pca_v1.json"))

The compute path then loads that artifact and applies it.

Lazy-imports ``sklearn.decomposition.PCA`` so the family itself
stays light-dependency: the test suite + production compute don't
need scikit-learn installed. Only the trainer does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from quant_platform.research.features.learned.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PCAArtifact,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def fit_pca_artifact(
    *,
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    n_components: int,
    family_version: str,
    drop_nan_rows: bool = True,
    standardise: bool = True,
    validate_against_registry: bool = False,
    extra_metadata: dict[str, str] | None = None,
) -> PCAArtifact:
    """Fit PCA on ``panel`` and return a frozen artifact.

    Parameters
    ----------
    panel:
        Training panel. Must carry every column in ``feature_names``.
        Extra columns are ignored.
    feature_names:
        Source feature column names. The artifact carries this list
        verbatim; the compute path validates that the inference
        panel includes every name before the matmul.
    n_components:
        Number of principal components to fit and emit. Must match
        the family's ``expected_n_components`` at compute time.
    family_version:
        Which learned-family version this artifact targets, e.g.
        ``"learned-representations-v1"``. Stamped onto the artifact
        for compatibility checks at compute time.
    drop_nan_rows:
        When ``True`` (default), drop rows with any NaN **or ±inf**
        in ``feature_names`` before fitting. PCA's covariance
        estimate is undefined under NaN, and sklearn rejects ±inf
        outright; ±inf is coerced to NaN first and then the same
        row-drop step removes it. The operator must explicitly opt
        in to a NaN/inf-tolerant fit by passing ``False`` (which
        will hit sklearn's behaviour, currently a hard error).
        Note: ±inf has been observed in upstream feature panels when
        ``safe_div`` edge cases slip through; this branch closes
        that gap so the trainer's contract is "give me anything you
        produced from the feature factory and I'll handle the
        not-finite rows."
    standardise:
        When ``True`` (default), standardise every source column to
        unit std-dev before fitting PCA, and bake the per-feature
        std-dev into :attr:`PCAArtifact.scale` so the compute path
        applies the same transform at inference. This is **the
        correct default** when the source panel mixes scales (e.g.
        ``dollar_volume_20d`` in the millions alongside ``ret_1d``
        near ``1e-2``); without it PCA collapses the entire variance
        budget onto the highest-scale column and the EVR distribution
        becomes uninformative (the 2026-05-26 backtest hit
        ``EVR == [1.000, 0.000, ..., 0.000]`` for this exact reason
        — see project_backtest_latest_stack).

        Pass ``False`` only when the caller has already standardised
        the panel out-of-band. In that case the artifact's
        :attr:`scale` is written as a tuple of ``1.0`` so the compute
        math stays uniform without changing behaviour.
    validate_against_registry:
        When ``True``, check that every name in ``feature_names``
        resolves to an entry in the process-global
        :class:`FeatureRegistry`. Catches typos in feature names and
        stale references to features that have been renamed by a
        family-version bump. Default ``False`` for backward-compat
        with operators who use the trainer on hand-curated panels
        that don't depend on the registry — flip to ``True`` for
        any production retraining workflow.
    extra_metadata:
        Free-form metadata to merge into :attr:`PCAArtifact.fit_metadata`
        on top of the automatic fields (date, sample size, sklearn
        version). The operator should include their training window
        bounds here for audit. **Values are coerced to strings** —
        see :class:`PCAArtifact` docstring for the string-only
        constraint.

    Raises
    ------
    ImportError
        When ``scikit-learn`` is not installed. Install with
        ``pip install scikit-learn`` — the family itself doesn't
        require it; only this trainer does.
    ValueError
        On shape / count mismatches that ``PCAArtifact.__post_init__``
        catches, or (when ``validate_against_registry=True``) when
        a name in ``feature_names`` doesn't resolve in the global
        registry, or (when ``standardise=True``) when any source
        column has zero std-dev (a constant column would divide-by-
        zero at standardisation time).
    """
    try:
        import sklearn
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "fit_pca_artifact requires scikit-learn. Install with "
            "`pip install scikit-learn`. The learned-representations-v1 "
            "feature family itself does NOT depend on sklearn — only "
            "this trainer does."
        ) from exc

    feature_names_tuple = tuple(feature_names)
    if not feature_names_tuple:
        raise ValueError("fit_pca_artifact: feature_names must be non-empty")

    missing = [col for col in feature_names_tuple if col not in panel.columns]
    if missing:
        raise ValueError(f"fit_pca_artifact: panel missing required feature columns: {missing!r}")

    if validate_against_registry:
        # Lazy import to avoid a circular at module load time. The
        # registry is the canonical source of truth for what feature
        # names are valid at training time. Names absent from the
        # registry are typos or stale references — catch them here
        # rather than at compute time months later.
        from quant_platform.research.features import get_global_registry

        registry = get_global_registry()
        unregistered = [name for name in feature_names_tuple if not registry.has(name)]
        if unregistered:
            raise ValueError(
                f"fit_pca_artifact: feature_names not present in the global "
                f"FeatureRegistry: {unregistered!r}. Either the names are typos "
                f"or the source family hasn't been imported yet. Call "
                f"``bootstrap_default_families()`` before fitting if you want "
                f"all shipped families discoverable."
            )

    matrix = panel[list(feature_names_tuple)]
    if drop_nan_rows:
        # Coerce ±inf -> NaN so the same dropna() step removes them.
        # Upstream feature panels occasionally carry ±inf from
        # safe_div edge cases on zero / near-zero denominators;
        # sklearn rejects ±inf with a hard error before fit, so
        # callers had to remember to clean inf themselves. Folding
        # it into the dropna branch keeps the trainer's contract
        # symmetric across both not-finite values.
        matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
    if len(matrix) < n_components:
        raise ValueError(
            f"fit_pca_artifact: insufficient samples after NaN-drop "
            f"({len(matrix)} rows) for n_components={n_components}"
        )

    raw_matrix = matrix.to_numpy(dtype=float)

    if standardise:
        # Detect constant columns BEFORE running StandardScaler. sklearn
        # silently rewrites a near-zero std-dev to 1.0 (so its
        # transform doesn't divide-by-zero), which would let a
        # constant column leak into the artifact as a feature with
        # scale==1 and zero information content — the operator
        # wouldn't notice until they wondered why PC_i had a constant
        # loading. Raising here surfaces the panel-cleanliness
        # problem at training time.
        #
        # We use peak-to-peak (max - min) rather than std because
        # std on a constant column carries floating-point noise of
        # order 1e-16 — that's > 0 but isn't real variance. The range
        # is exactly 0 for a genuinely-constant column at IEEE 754
        # precision and gives a clean threshold.
        raw_range = raw_matrix.max(axis=0) - raw_matrix.min(axis=0)
        zero_scale_cols = [feature_names_tuple[i] for i, r in enumerate(raw_range) if r == 0.0]
        if zero_scale_cols:
            raise ValueError(
                f"fit_pca_artifact: source columns have zero variance and "
                f"cannot be standardised: {zero_scale_cols!r}. Drop the "
                "constant column(s) from the training panel, or pass "
                "``standardise=False`` if the operator has already handled "
                "scaling out-of-band."
            )
        scaler = StandardScaler()
        # Use scaler.fit so we get the per-feature mean and std-dev to
        # bake into the artifact. We pass the standardised matrix to
        # PCA below; PCA's own mean_ will then be (approximately) zero,
        # but we still record the raw-space mean so the compute path
        # can centre raw rows before dividing by scale.
        scaler.fit(raw_matrix)
        raw_mean = np.asarray(scaler.mean_, dtype=float)
        raw_scale = np.asarray(scaler.scale_, dtype=float)
        standardised = (raw_matrix - raw_mean[None, :]) / raw_scale[None, :]
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(standardised)
        artifact_mean = raw_mean
        artifact_scale = raw_scale
    else:
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(raw_matrix)
        artifact_mean = np.asarray(pca.mean_, dtype=float)
        # Write all-ones into scale so the compute path's
        # ``(matrix - mean) / scale`` reduces to ``matrix - mean``
        # — identical to the pre-v2 behaviour for callers who have
        # already standardised out-of-band.
        artifact_scale = np.ones_like(artifact_mean)

    metadata = {
        "fit_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_samples_fit": str(len(matrix)),
        "sklearn_version": str(sklearn.__version__),
        "standardised": str(bool(standardise)),
    }
    if extra_metadata is not None:
        for k, v in extra_metadata.items():
            metadata[str(k)] = str(v)

    return PCAArtifact(
        artifact_version=ARTIFACT_SCHEMA_VERSION,
        family_version=family_version,
        n_components=int(n_components),
        feature_names=feature_names_tuple,
        mean=tuple(float(v) for v in artifact_mean),
        scale=tuple(float(v) for v in artifact_scale),
        components=tuple(
            tuple(float(v) for v in row) for row in np.asarray(pca.components_, dtype=float)
        ),
        explained_variance_ratio=tuple(
            float(v) for v in np.asarray(pca.explained_variance_ratio_, dtype=float)
        ),
        fit_metadata=metadata,
    )


__all__ = ["fit_pca_artifact"]
