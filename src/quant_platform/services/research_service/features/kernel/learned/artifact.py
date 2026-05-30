"""Frozen PCA artifact for the ``learned-representations-v1`` family.

The :class:`PCAArtifact` is the **single source of truth** for the
learned family's transform parameters. It is emitted by an
**out-of-band trainer** (see :mod:`.trainer`) and persisted to disk
as JSON. The feature-compute path then loads it and applies a pure
deterministic transform — no fitting happens inside the family at
compute time.

Why a separate frozen artifact (not in-line fitting)
----------------------------------------------------

* **PIT safety.** A trainer fitting PCA during compute would leak
  forward information across the walk-forward fold boundary unless
  the trainer is fold-aware. Forcing the operator to emit a *frozen*
  artifact tied to a specific training period eliminates the
  question.
* **Determinism.** The compute path is a pure matmul + L2 norm. Same
  input × same artifact = same output, byte-for-byte.
* **Reproducibility.** The artifact carries fit metadata (source
  feature list, sample size, fit date) so the same artifact can be
  loaded years later and the transform behaves identically.
* **Governance.** Promoting a learned representation requires a
  separate, auditable artifact-release step — not a code change.

Storage format
--------------

All numeric fields are stored as **tuples of floats** (not numpy
arrays). This makes the dataclass truly frozen (a frozen dataclass
can still hold mutable ndarrays; tuples can't be mutated), and
guarantees the JSON serialisation is lossless and round-trips
exactly.

The compute path lifts the tuples back into ``np.ndarray`` via the
:meth:`as_numpy` helper for matmul performance.

:attr:`fit_metadata` field constraint
-------------------------------------

The metadata map is typed ``dict[str, str]`` and the trainer
coerces every value through ``str(...)`` at write time. This is
intentional: it makes the artifact's JSON output unambiguous
(every value round-trips as a string, no parser-side type
guessing) and gives auditors a flat human-readable provenance log.

If an operator wants to store numeric metadata (e.g. R² of the fit,
sample size in millions), the value is preserved as a string —
consumers must ``float(...)`` or ``int(...)`` parse it back. This
asymmetry is documented here so an operator who stores
``"r_squared": 0.847`` doesn't expect a round-tripped Python float.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Bumped when the artifact dataclass schema changes (NOT when an
#: operator retrains and emits a new artifact under the same schema).
#: Persisted artifacts carry this value so the loader can reject
#: incompatible payloads.
#:
#: ``v2`` (2026-05-26) added the :attr:`PCAArtifact.scale` field so the
#: compute path can apply the trainer's per-feature standardisation
#: before the projection matmul. ``v1`` artifacts have no scale and
#: are rejected by the loader — re-emit through
#: :func:`~.trainer.fit_pca_artifact` to upgrade.
ARTIFACT_SCHEMA_VERSION: str = "pca-artifact-v2"


@dataclass(frozen=True)
class PCAArtifact:
    """A frozen, JSON-serialisable PCA transform.

    Attributes
    ----------
    artifact_version:
        :data:`ARTIFACT_SCHEMA_VERSION` at write time. The loader
        rejects payloads whose version doesn't match.
    family_version:
        Which learned-family version this artifact targets, e.g.
        ``"learned-representations-v1"``. The compute path checks
        that ``config.version == artifact.family_version`` so an
        artifact trained against a different family version can't
        silently be applied.
    n_components:
        Number of PCs the artifact produces. Must equal
        ``LearnedConfig.expected_n_components`` at compute time.
    feature_names:
        Source feature column names in the order PCA was fit on.
        The compute path validates that the input panel carries
        every name and aligns columns in this order before the
        matmul.
    mean:
        Per-feature mean used for centering. Length = len(feature_names).
    scale:
        Per-feature std-dev used for standardisation. Length =
        len(feature_names). The compute path divides the centred row
        by this vector elementwise before the matmul, so that PCA
        operates on a unit-variance representation regardless of the
        absolute scale of the source features. When the trainer is
        run with ``standardise=False`` (see
        :func:`~.trainer.fit_pca_artifact`), the trainer writes a
        tuple of ``1.0`` here so the compute math stays uniform
        without changing behaviour. Every entry must be strictly
        positive — a zero would otherwise divide-by-zero on the
        first compute call.
    components:
        PCA loading matrix shape (n_components, n_features). Stored
        as a tuple-of-tuples; the compute helper rebuilds it as a
        2D ndarray via :meth:`as_numpy`.
    explained_variance_ratio:
        Per-component fraction of total variance. Length = n_components.
        Carried for diagnostics; not consumed by the compute path.
    fit_metadata:
        Free-form string-keyed map for fit provenance: training
        date, sample size, source families, sklearn version, etc.
        Persisted so future auditors can re-derive the artifact.
    """

    artifact_version: str
    family_version: str
    n_components: int
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    fit_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"PCAArtifact.artifact_version must equal "
                f"{ARTIFACT_SCHEMA_VERSION!r}; got {self.artifact_version!r}"
            )
        if not self.family_version.strip():
            raise ValueError("PCAArtifact.family_version must be non-empty")
        if self.n_components <= 0:
            raise ValueError(f"PCAArtifact.n_components must be > 0; got {self.n_components}")
        if not self.feature_names:
            raise ValueError("PCAArtifact.feature_names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("PCAArtifact.feature_names must be unique")

        n_features = len(self.feature_names)
        if len(self.mean) != n_features:
            raise ValueError(
                f"PCAArtifact.mean must have length {n_features}; got {len(self.mean)}"
            )
        if len(self.scale) != n_features:
            raise ValueError(
                f"PCAArtifact.scale must have length {n_features}; got {len(self.scale)}"
            )
        for i, s in enumerate(self.scale):
            if not (s > 0.0):
                raise ValueError(
                    f"PCAArtifact.scale[{i}] (feature {self.feature_names[i]!r}) "
                    f"must be > 0; got {s!r}. A non-positive scale would "
                    "divide-by-zero at compute time — re-emit the artifact, "
                    "and check the trainer panel for constant columns."
                )
        if len(self.components) != self.n_components:
            raise ValueError(
                f"PCAArtifact.components must have {self.n_components} rows; "
                f"got {len(self.components)}"
            )
        for i, row in enumerate(self.components):
            if len(row) != n_features:
                raise ValueError(
                    f"PCAArtifact.components row {i} must have length {n_features}; got {len(row)}"
                )
        if len(self.explained_variance_ratio) != self.n_components:
            raise ValueError(
                f"PCAArtifact.explained_variance_ratio must have length "
                f"{self.n_components}; got {len(self.explained_variance_ratio)}"
            )

    # ------------------------------------------------------------------
    # ndarray accessors — used by the compute path; allocated fresh on
    # each call so the frozen dataclass's tuples stay the canonical
    # source of truth.
    # ------------------------------------------------------------------

    def components_as_numpy(self) -> np.ndarray:
        """Return ``components`` as a 2D float array of shape
        ``(n_components, n_features)``.

        **A fresh array is allocated on every call** — the tuple
        storage is the canonical source of truth and we don't cache
        an ndarray copy on the frozen dataclass (it would defeat the
        true-immutability guarantee). For the compute path this is
        fine; the conversion happens once per
        :func:`compute_learned_features` invocation. Don't call this
        method in a tight loop on the same artifact — capture the
        return value locally instead.
        """
        return np.asarray(self.components, dtype=float)

    def mean_as_numpy(self) -> np.ndarray:
        """Return ``mean`` as a 1D float array. Same allocation
        semantics as :meth:`components_as_numpy` — fresh array per
        call, no caching."""
        return np.asarray(self.mean, dtype=float)

    def scale_as_numpy(self) -> np.ndarray:
        """Return ``scale`` as a 1D float array. Same allocation
        semantics as :meth:`components_as_numpy` — fresh array per
        call, no caching."""
        return np.asarray(self.scale, dtype=float)

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict.

        Uses :func:`dataclasses.asdict` which recursively converts
        nested tuples to lists — that's intentional and necessary
        for JSON serialisation. The asymmetry with :meth:`from_dict`
        (which reads lists and reconstructs tuples) keeps the
        canonical in-memory representation immutable while the
        on-disk format stays standard JSON.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PCAArtifact:
        """Construct from a parsed JSON payload. Rejects payloads
        whose ``artifact_version`` doesn't match the current schema."""
        version = payload.get("artifact_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"PCAArtifact.from_dict: unsupported artifact_version "
                f"{version!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        return cls(
            artifact_version=str(version),
            family_version=str(payload["family_version"]),
            n_components=int(payload["n_components"]),
            feature_names=tuple(payload["feature_names"]),
            mean=tuple(float(v) for v in payload["mean"]),
            scale=tuple(float(v) for v in payload["scale"]),
            components=tuple(tuple(float(v) for v in row) for row in payload["components"]),
            explained_variance_ratio=tuple(float(v) for v in payload["explained_variance_ratio"]),
            fit_metadata={str(k): str(v) for k, v in payload.get("fit_metadata", {}).items()},
        )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PCAArtifact",
]
