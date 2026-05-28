"""Feature-family plugin contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quant_platform.core.domain.research import FeatureRequest, FeatureResult


class NamedPlugin(Protocol):
    """Minimal plugin identity contract."""

    @property
    def name(self) -> str:
        """Stable feature-family key."""
        ...


@runtime_checkable
class FeatureComputer(Protocol):
    """Registered computer for one feature family and feature-set version."""

    @property
    def feature_family(self) -> str:
        """Feature family name, such as ``ohlcv`` or ``text``."""
        ...

    @property
    def feature_set_version(self) -> str:
        """Version emitted by this computer."""
        ...

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Input datasets or feature sets required by this computer."""
        ...

    @property
    def output_features(self) -> tuple[str, ...]:
        """Ordered output feature names used for schema hashing."""
        ...

    @property
    def schema_hash(self) -> str:
        """Schema hash for ``output_features``."""
        ...

    async def compute(self, request: FeatureRequest) -> FeatureResult:
        """Compute and return typed feature vectors."""
        ...


class FeatureFamilyDescriptor(NamedPlugin, Protocol):
    """Metadata-only feature-family descriptor for discovery catalogs.

    A descriptor advertises a feature family but does *not* promise any
    executable computers.  Discovery surfaces (the plugin catalog) hold
    descriptors; only :class:`FeatureFamilyPlugin` may be wired into the
    runtime ``FeatureFamilyRegistry``.
    """

    @property
    def feature_set_version(self) -> str:
        """Version emitted by the feature family."""
        ...

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Input feature/source names required to compute this family."""
        ...


class FeatureFamilyPlugin(FeatureFamilyDescriptor, Protocol):
    """Executable feature-family extension point.

    Extends :class:`FeatureFamilyDescriptor` with the runtime contract: a
    plugin must be able to build at least one concrete feature computer.
    """

    def build_computers(self) -> tuple[FeatureComputer, ...]:
        """Return registered computers for this family/version."""
        ...


__all__ = [
    "FeatureComputer",
    "FeatureFamilyDescriptor",
    "FeatureFamilyPlugin",
    "NamedPlugin",
]
