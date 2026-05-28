"""Process-wide feature specification registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from quant_platform.research.features.contracts import (
        FamilyManifest,
        FeatureFamily,
        FeatureSpec,
    )

FeatureKey = tuple[str, str]


class FeatureRegistry:
    """Mutable registry keyed by ``(feature name, feature version)``.

    The registry allows historical versions of the same named feature to
    coexist. Callers that need the current catalog can use ``get_latest`` or
    ``get(name)``; callers replaying evidence should pass the explicit version.
    """

    def __init__(self) -> None:
        self._by_key: dict[FeatureKey, FeatureSpec] = {}
        self._latest_version_by_name: dict[str, str] = {}
        self._families_by_key: dict[FeatureKey, FamilyManifest] = {}

    def register(self, spec: FeatureSpec) -> None:
        """Register one spec.

        Re-registering an identical ``(name, version)`` pair is idempotent.
        Re-registering a different spec under the same pair fails closed.
        """
        key = (spec.name, spec.version)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing != spec:
                raise ValueError(
                    "FeatureRegistry: feature "
                    f"{spec.name!r} version={spec.version!r} already registered "
                    "with a different spec"
                )
            return
        self._by_key[key] = spec
        self._latest_version_by_name[spec.name] = spec.version

    def register_many(self, specs: Iterable[FeatureSpec]) -> None:
        """Register every spec in ``specs``."""
        for spec in specs:
            self.register(spec)

    def get(self, name: str, version: str | None = None) -> FeatureSpec:
        """Return a registered spec by name and optional version."""
        if version is None:
            return self.get_latest(name)
        key = (name, version)
        try:
            return self._by_key[key]
        except KeyError as exc:
            known = ", ".join(f"{n}@{v}" for n, v in self._sorted_keys())
            raise KeyError(
                f"FeatureRegistry: no feature registered as {name!r} "
                f"version={version!r}; known features: {known}"
            ) from exc

    def get_latest(self, name: str) -> FeatureSpec:
        """Return the most recently registered version for ``name``."""
        try:
            version = self._latest_version_by_name[name]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(
                f"FeatureRegistry: no feature registered as {name!r}; known features: {known}"
            ) from exc
        return self._by_key[(name, version)]

    def has(self, name: str, version: str | None = None) -> bool:
        """Return whether the registry contains ``name`` and optional version."""
        if version is None:
            return name in self._latest_version_by_name
        return (name, version) in self._by_key

    def all(self) -> tuple[FeatureSpec, ...]:
        """Return every registered spec, sorted by ``(name, version)``."""
        return tuple(self._by_key[key] for key in self._sorted_keys())

    def names(self) -> tuple[str, ...]:
        """Return unique registered feature names."""
        return tuple(sorted(self._latest_version_by_name))

    def by_family(self, family: FeatureFamily) -> tuple[FeatureSpec, ...]:
        """Return every spec whose ``family`` matches ``family``."""
        return tuple(spec for spec in self.all() if spec.family == family)

    def by_version(self, version: str) -> tuple[FeatureSpec, ...]:
        """Return every spec whose ``version`` matches ``version``."""
        return tuple(spec for spec in self.all() if spec.version == version)

    # ---- Family-level indexing -------------------------------------------

    def register_family(self, manifest: FamilyManifest) -> None:
        """Register a :class:`FamilyManifest`.

        Also registers every spec in :attr:`FamilyManifest.feature_specs`
        through :meth:`register_many`, so callers do not have to make two
        calls. Re-registering the same ``(name, version)`` pair with an
        identical manifest is idempotent; re-registering with a different
        manifest fails closed.
        """
        key = (manifest.name, manifest.version)
        existing = self._families_by_key.get(key)
        if existing is not None:
            if existing != manifest:
                raise ValueError(
                    f"FeatureRegistry: family {manifest.name!r} version="
                    f"{manifest.version!r} already registered with a "
                    "different manifest"
                )
            return
        self.register_many(manifest.feature_specs)
        self._families_by_key[key] = manifest

    def get_family(self, name: str, version: str | None = None) -> FamilyManifest:
        """Return a registered family manifest by name and optional version."""
        if version is None:
            candidates = [(k, m) for k, m in self._families_by_key.items() if k[0] == name]
            if not candidates:
                raise KeyError(
                    f"FeatureRegistry: no family registered as {name!r}; "
                    f"known families: {self.families()!r}"
                )
            # Return the most-recently-registered version (last-write-wins).
            return candidates[-1][1]
        try:
            return self._families_by_key[(name, version)]
        except KeyError as exc:
            raise KeyError(
                f"FeatureRegistry: no family registered as {name!r} version="
                f"{version!r}; known families: {self.families()!r}"
            ) from exc

    def families(self) -> tuple[str, ...]:
        """Return unique registered family names (sorted)."""
        return tuple(sorted({k[0] for k in self._families_by_key}))

    def has_family(self, name: str, version: str | None = None) -> bool:
        """Return whether a family is registered."""
        if version is None:
            return any(k[0] == name for k in self._families_by_key)
        return (name, version) in self._families_by_key

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.all())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)

    def _sorted_keys(self) -> tuple[FeatureKey, ...]:
        return tuple(sorted(self._by_key))


_GLOBAL_REGISTRY: FeatureRegistry = FeatureRegistry()


def get_global_registry() -> FeatureRegistry:
    """Return the module-global feature registry."""
    return _GLOBAL_REGISTRY


def register(spec: FeatureSpec) -> None:
    """Register ``spec`` in the global registry."""
    _GLOBAL_REGISTRY.register(spec)


def register_many(specs: Iterable[FeatureSpec]) -> None:
    """Register every entry of ``specs`` in the global registry."""
    _GLOBAL_REGISTRY.register_many(specs)


def register_family(manifest: FamilyManifest) -> None:
    """Register ``manifest`` in the global registry."""
    _GLOBAL_REGISTRY.register_family(manifest)


__all__ = [
    "FeatureKey",
    "FeatureRegistry",
    "get_global_registry",
    "register",
    "register_family",
    "register_many",
]
