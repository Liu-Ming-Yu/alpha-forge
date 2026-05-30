"""Configuration for the ``learned-representations-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.services.research_service.features.kernel.contracts import BaseFamilyConfig

#: Family version. Bump on formula change, n_components change, or
#: reconstruction-error definition change. v1 is the artifact-backed
#: PCA representation: 8 PCs + 1 reconstruction error.
FEATURE_SET_VERSION: str = "learned-representations-v1"

#: The fixed number of PCs the v1 family emits. The artifact's
#: ``n_components`` must equal this value at compute time — an
#: operator who needs a different component count must bump the
#: family version (e.g. ``learned-representations-v2`` with 12 PCs).
DEFAULT_N_COMPONENTS: int = 8


@dataclass(frozen=True)
class LearnedConfig(BaseFamilyConfig):
    """Frozen config for the learned-representations feature factory.

    Attributes
    ----------
    version:
        Family-version string stamped into every emitted FeatureSpec.
        The compute path also checks that the supplied
        :class:`PCAArtifact.family_version` matches — applying an
        artifact trained for v2 under v1 is rejected.
    expected_n_components:
        Hard invariant on the artifact's ``n_components``. Default
        :data:`DEFAULT_N_COMPONENTS` (8). The number appears in
        feature column names (``learned_pc_1`` .. ``learned_pc_8``),
        so changing it requires a family-version bump.
    """

    version: str = FEATURE_SET_VERSION
    expected_n_components: int = DEFAULT_N_COMPONENTS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expected_n_components <= 0:
            raise ValueError("LearnedConfig.expected_n_components must be > 0")


DEFAULT_CONFIG: LearnedConfig = LearnedConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_N_COMPONENTS",
    "FEATURE_SET_VERSION",
    "LearnedConfig",
]
