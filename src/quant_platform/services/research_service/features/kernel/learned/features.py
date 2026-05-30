"""``learned-representations-v1`` feature factory.

Nine features:

* ``learned_pc_1`` .. ``learned_pc_8`` — 8 principal-component scores
  produced by projecting the centred source-feature row onto the
  artifact's PCA components.
* ``learned_reconstruction_error`` — per-row L2 norm of the residual
  ``(x − x_hat)`` where ``x_hat`` is the PCA reconstruction. High
  values are rows the artifact's principal subspace doesn't explain
  well — anomaly / regime-shift detector for free.

The family is **representation-only**. No PCA fitting happens inside
:func:`compute_learned_features`; the artifact is fit out-of-band by
:func:`~.trainer.fit_pca_artifact` and persisted to disk as JSON
via :func:`~.loader.save_pca_artifact`. The compute function
validates the artifact, lifts its tuples into ndarrays, and does
**one matmul + one L2 norm per row**. Same input × same artifact =
same output, byte-for-byte.

Direction conventions and evidence gating
-----------------------------------------

All nine features ship ``expected_direction="unknown"`` and
``larger_is_better=False``. The principal-component axes are
orthogonal mathematical constructs with no inherent direction — a
sign flip on the underlying PCA loadings reverses every score's
sign without changing its information content. Reconstruction error
is genuinely directional (low = expected; high = anomaly) but its
predictive direction on forward returns is empirically uncertain.
Promotion to a directional spec is a family-version bump.

required_inputs convention (DELIBERATE EXCEPTION)
-------------------------------------------------

Each spec carries an empty ``required_inputs`` tuple. **This is a
deliberate exception** from the rest of the catalogue, where every
spec lists the raw input columns it depends on. The TRUE input
list for this family is the artifact's
:attr:`PCAArtifact.feature_names`, which the compute path validates
against the input panel at the boundary.

Why the exception:

* The artifact's ``feature_names`` carries the real dependency list,
  and it can vary across artifacts (one operator may train on 147
  features from all 9 prior families; another may train on a subset).
  Hard-coding a 147-entry ``required_inputs`` on every spec would
  pin the family to a specific source catalogue and force a
  family-version bump every time any source family bumps.
* The compute-time validation (:func:`_validate_panel_has_artifact_features`)
  catches missing columns explicitly at the panel boundary, so the
  contract is enforced — just at a different layer than the
  ``required_inputs`` tuple normally provides.

Consequence: **the platform's dependency-graph tooling (if/when it
runs over `FeatureSpec.required_inputs`) must special-case this
family**. A future v2 may introduce a ``required_artifact: bool``
field on FeatureSpec so the dependency graph can signal "needs an
artifact, not direct columns."

"Registered" does NOT mean "ready to compute"
---------------------------------------------

This family registers its MANIFEST at process-import time — the
nine ``learned_*`` features are in :class:`FeatureRegistry` even
before any artifact is loaded. ``FeatureRegistry.has("learned_pc_1",
"learned-representations-v1")`` returns ``True`` immediately on
import; that is correct (the spec exists) but does NOT imply the
family can compute. A caller that invokes
:func:`compute_learned_features` without first loading an artifact
gets a clean "missing source feature columns" or
``_validate_artifact_compatibility`` error — never a silent partial
result. Operators using the registry to schedule compute should
also call :func:`~.loader.load_pca_artifact` and confirm it
succeeds before assuming the family is usable.

NaN handling
------------

Any source feature missing from the panel raises at the boundary
(operator must include all source columns). For row-level NaN
(present column with NaN cell), the compute function emits NaN for
all nine output features on that row. This is the conservative
choice: don't impute, don't drop, just propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureFrame,
    FeatureSpec,
)
from quant_platform.services.research_service.features.kernel.learned.config import (
    DEFAULT_CONFIG,
    LearnedConfig,
)
from quant_platform.services.research_service.features.kernel.transforms import DEFAULT_KEY_COLUMNS

if TYPE_CHECKING:
    from quant_platform.services.research_service.features.kernel.learned.artifact import (
        PCAArtifact,
    )


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


def _build_specs(version: str, *, n_components: int) -> tuple[FeatureSpec, ...]:
    pc_specs = tuple(
        FeatureSpec(
            name=f"learned_pc_{i}",
            family="learned",
            description=(
                f"Principal component {i} of the source-feature panel, "
                "projected onto the loaded PCA artifact's components. "
                "Sign-invariant: the artifact's underlying SVD pinning "
                "determines the orientation of the axis, not a directional "
                "claim about the feature. Direction is empirically "
                "uncertain and the spec stays evidence-gated."
            ),
            expected_direction="unknown",
            required_inputs=(),  # See module docstring — artifact carries the true dep list.
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        )
        for i in range(1, n_components + 1)
    )
    recon_spec = FeatureSpec(
        name="learned_reconstruction_error",
        family="learned",
        description=(
            "Per-row L2 norm of the residual (x − x_hat), where x_hat is "
            "the PCA reconstruction from the loaded artifact. High values = "
            "rows the artifact's principal subspace doesn't explain well, "
            "i.e. potential anomalies or regime-shift days. Non-negative "
            "by construction; NaN when any source feature in the row is NaN. "
            "**SCALE-DEPENDENT.** The norm is unnormalised, so its magnitude "
            "scales with the absolute scale of the source features. If the "
            "source panel mixes scales (e.g. revenue in millions alongside "
            "ratios in tenths), the error is dominated by the highest-scale "
            "feature's residual. Operators using this feature for "
            "cross-instrument comparison should either (a) pre-normalise "
            "source features before fitting the artifact, or (b) consume "
            "the feature per-instrument-rank rather than per-instrument "
            "level. A v2 of this family will likely add "
            "`learned_relative_reconstruction_error` = norm(residual) / "
            "norm(x − mean) once walk-forward evidence shows the relative "
            "metric carries information the absolute one doesn't."
        ),
        expected_direction="unknown",
        required_inputs=(),
        point_in_time=True,
        lookback_days=1,
        version=version,
        larger_is_better=False,
    )
    return (*pc_specs, recon_spec)


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version, n_components=DEFAULT_CONFIG.expected_n_components
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


def _specs_for_config(config: LearnedConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.expected_n_components == DEFAULT_CONFIG.expected_n_components
    ):
        return FEATURE_SPECS
    return _build_specs(config.version, n_components=config.expected_n_components)


# ---------------------------------------------------------------------------
# Compute — deterministic transform, NO fitting
# ---------------------------------------------------------------------------


def _validate_artifact_compatibility(
    artifact: PCAArtifact,
    config: LearnedConfig,
) -> None:
    """Refuse to compute when the artifact and config disagree.

    Catches: an artifact trained for a different family version
    (v2-shaped components applied under v1), or an artifact whose
    n_components doesn't match the family's spec-emitted column
    count.
    """
    if artifact.family_version != config.version:
        raise ValueError(
            f"PCAArtifact.family_version={artifact.family_version!r} does not "
            f"match LearnedConfig.version={config.version!r}. An artifact "
            "trained for a different family version cannot be applied here — "
            "re-emit the artifact under the correct version, or bump the "
            "family-version field."
        )
    if artifact.n_components != config.expected_n_components:
        raise ValueError(
            f"PCAArtifact.n_components={artifact.n_components} does not match "
            f"LearnedConfig.expected_n_components={config.expected_n_components}. "
            f"The v1 family emits exactly {config.expected_n_components} PC "
            "columns; train an artifact with the matching n_components."
        )


def _validate_panel_has_artifact_features(panel: pd.DataFrame, artifact: PCAArtifact) -> None:
    """Boundary validator: the panel must carry every source feature
    the artifact was trained on, in any order."""
    missing = [col for col in artifact.feature_names if col not in panel.columns]
    if missing:
        raise ValueError(
            f"compute_learned_features: panel missing source feature columns "
            f"required by the artifact: {missing!r}"
        )


def compute_learned_features(
    *,
    panel: pd.DataFrame,
    artifact: PCAArtifact,
    config: LearnedConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Apply the PCA artifact to ``panel`` and return the learned features.

    The transform per row is:

    1. ``z = (x - artifact.mean) / artifact.scale`` — standardise into
       the artifact's training-time unit-variance space.
    2. ``pc_scores = z @ components.T`` — project onto the artifact's
       principal subspace.
    3. ``residual_z = z - pc_scores @ components`` — orthogonal
       complement in standardised space.
    4. ``learned_reconstruction_error = ||residual_z||_2`` — per-row
       L2 norm of the residual.

    ``artifact.scale`` is a tuple of ``1.0`` for artifacts emitted with
    ``standardise=False`` (or for any pre-v2 callers that pre-standardised
    out-of-band and roundtrip through the trainer), so steps 1–4 reduce
    to the pre-v2 centre-and-project math in that mode.

    Parameters
    ----------
    panel:
        Long-format DataFrame keyed by ``(instrument_id, date)``.
        Must carry every column in ``artifact.feature_names``.
        Extra columns are ignored.
    artifact:
        Frozen :class:`PCAArtifact` (already loaded via
        :func:`~.loader.load_pca_artifact` or constructed in tests).
    config:
        :class:`LearnedConfig`. The compute path validates that the
        config and artifact agree on family version + n_components.

    Returns
    -------
    FeatureFrame
        9 columns: ``learned_pc_1`` .. ``learned_pc_8`` +
        ``learned_reconstruction_error``.
    """
    _validate_artifact_compatibility(artifact, config)
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}

    if panel.empty:
        empty = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "date": pd.Series(dtype="datetime64[ns]"),
                **{name: pd.Series(dtype=float) for name in feature_names},
            }
        )
        return FeatureFrame(
            frame=empty,
            feature_names=feature_names,
            feature_specs=spec_by_name,
            coverage={name: 0 for name in feature_names},
            key_columns=DEFAULT_KEY_COLUMNS,
        )

    _validate_panel_has_artifact_features(panel, artifact)

    # Source matrix in the artifact's column order.
    source_columns = list(artifact.feature_names)
    matrix = panel[source_columns].to_numpy(dtype=float)
    mean = artifact.mean_as_numpy()
    scale = artifact.scale_as_numpy()
    components = artifact.components_as_numpy()
    n_components = artifact.n_components

    # Identify NaN-containing rows; we'll mask them to NaN at the end.
    nan_mask = np.isnan(matrix).any(axis=1)

    # Center, standardise, and project. For NaN-containing rows we set
    # the input values to zero (a placeholder) and overwrite the
    # output with NaN below — this avoids NaN propagating through the
    # matmul, which would otherwise make every PC score NaN for the
    # whole panel on some BLAS implementations.
    #
    # NOTE: the matmul output on NaN-masked rows is computed but
    # immediately discarded. We accept the wasted FLOPs because the
    # alternative (slicing the panel to non-NaN rows, projecting,
    # scattering back) costs Python-level index arithmetic that's
    # slower than a single vectorised matmul on the full panel for
    # any realistic NaN rate.
    safe_matrix = matrix.copy()
    safe_matrix[nan_mask] = 0.0
    # ``standardised`` is the v2 input to the projection: same row in
    # the artifact's unit-variance training-time space. For pre-
    # standardised artifacts (scale == 1) this reduces to the v1
    # mean-centred matrix.
    standardised = (safe_matrix - mean[None, :]) / scale[None, :]
    pc_scores = standardised @ components.T  # shape: (n_rows, n_components)
    # Reconstruction error is computed in **standardised space** — the
    # natural metric once features are unit-variance. For pre-v2
    # callers with scale == 1, this is identical to the old raw-space
    # residual (since ``raw - (pc @ comp + mean) == (raw - mean) - pc @ comp``).
    reconstructed_z = pc_scores @ components
    residual_z = standardised - reconstructed_z
    # On NaN rows the residual carries NaN through ``standardised``; we
    # explicitly zero the masked rows so np.linalg.norm doesn't
    # propagate NaN, then overwrite the output entries with NaN — the
    # temporary zeros never escape.
    safe_residual = residual_z.copy()
    safe_residual[nan_mask] = 0.0
    reconstruction_error = np.linalg.norm(safe_residual, axis=1)

    # Mask NaN rows on the final outputs.
    pc_scores[nan_mask] = np.nan
    reconstruction_error[nan_mask] = np.nan

    # Assemble output.
    output_cols: dict[str, object] = {
        "instrument_id": panel["instrument_id"].to_numpy(),
        "date": panel["date"].to_numpy(),
    }
    for i in range(n_components):
        output_cols[f"learned_pc_{i + 1}"] = pc_scores[:, i]
    output_cols["learned_reconstruction_error"] = reconstruction_error

    output = pd.DataFrame(output_cols)
    output[list(feature_names)] = output[list(feature_names)].replace([np.inf, -np.inf], np.nan)
    coverage = {name: int(output[name].notna().sum()) for name in feature_names}
    return FeatureFrame(
        frame=output,
        feature_names=feature_names,
        feature_specs=spec_by_name,
        coverage=coverage,
        key_columns=DEFAULT_KEY_COLUMNS,
    )


__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_learned_features",
]
