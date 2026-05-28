"""Typed feature-family registry and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.core.domain.research import FeatureRequest, FeatureResult, FeatureVector

if TYPE_CHECKING:
    from quant_platform.core.contracts import FeatureComputer, FeatureFamilyPlugin


def feature_plugin_key(plugin: FeatureFamilyPlugin) -> str:
    """Return the stable feature plugin key for a family/version pair."""
    return f"{plugin.name}:{plugin.feature_set_version}"


@dataclass(frozen=True)
class _RegisteredFamily:
    """A feature-family plugin with computers resolved at registration time."""

    plugin: FeatureFamilyPlugin
    computers: tuple[FeatureComputer, ...]


@dataclass(frozen=True)
class FeatureFamilyRegistry:
    """Registry for versioned, executable feature-family plugins."""

    _families: dict[str, _RegisteredFamily] = field(default_factory=dict)

    @classmethod
    def from_plugins(cls, plugins: tuple[FeatureFamilyPlugin, ...]) -> FeatureFamilyRegistry:
        registry = cls()
        for plugin in plugins:
            registry.register(plugin)
        return registry

    def register(self, plugin: FeatureFamilyPlugin) -> None:
        """Register one family/version plugin."""
        if not plugin.name.strip():
            raise ValueError("feature family name must not be empty")
        if not plugin.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        key = feature_plugin_key(plugin)
        if key in self._families:
            raise ValueError(f"duplicate feature family plugin {key!r}")
        computers = plugin.build_computers()
        self._families[key] = _RegisteredFamily(plugin=plugin, computers=computers)

    def get(self, *, feature_family: str, feature_set_version: str) -> FeatureFamilyPlugin:
        """Return a registered family/version plugin."""
        key = f"{feature_family}:{feature_set_version}"
        try:
            return self._families[key].plugin
        except KeyError as exc:
            valid = ", ".join(self.keys())
            raise ValueError(f"unknown feature family {key!r}; valid families: {valid}") from exc

    def keys(self) -> tuple[str, ...]:
        """Return registered feature family keys."""
        return tuple(sorted(self._families))

    def family_for_version(self, feature_set_version: str) -> str | None:
        """Return the unique family name that owns a feature-set version.

        Returns ``None`` when no registered family produces the version, and
        raises when more than one family claims it (an ambiguous dispatch).
        """
        families = sorted(
            {
                family.plugin.name
                for family in self._families.values()
                if family.plugin.feature_set_version == feature_set_version
            }
        )
        if not families:
            return None
        if len(families) > 1:
            raise ValueError(
                f"feature_set_version {feature_set_version!r} is ambiguous "
                f"across families {families}"
            )
        return families[0]

    async def compute(
        self,
        *,
        feature_family: str,
        request: FeatureRequest,
    ) -> FeatureResult:
        """Compute a typed feature result through registered computers."""
        registered = self._families.get(f"{feature_family}:{request.feature_set_version}")
        if registered is None:
            matching_versions = tuple(
                sorted(
                    family.plugin.feature_set_version
                    for family in self._families.values()
                    if family.plugin.name == feature_family
                )
            )
            if matching_versions:
                return _failed_result(
                    request,
                    "feature_set_version_mismatch",
                    detail=(f"{request.feature_set_version!r} not in {matching_versions!r}"),
                    feature_family=feature_family,
                )
            self.get(feature_family=feature_family, feature_set_version=request.feature_set_version)
            raise AssertionError("unreachable")
        plugin = registered.plugin
        computers = registered.computers
        if not computers:
            return _failed_result(
                request,
                "feature_family_has_no_computers",
                detail=(
                    f"{feature_family}:{request.feature_set_version} has no registered computers"
                ),
                feature_family=feature_family,
            )
        missing_inputs = _missing_required_inputs(plugin.required_inputs, request)
        if missing_inputs:
            return _failed_result(
                request,
                "feature_required_inputs_missing",
                detail=f"missing required inputs: {', '.join(missing_inputs)}",
                feature_family=feature_family,
                extra={"missing_inputs": missing_inputs},
            )
        vectors: list[FeatureVector] = []
        for computer in computers:
            missing_inputs = _missing_required_inputs(computer.required_inputs, request)
            if missing_inputs:
                return _failed_result(
                    request,
                    "feature_required_inputs_missing",
                    detail=f"missing required inputs: {', '.join(missing_inputs)}",
                    feature_family=feature_family,
                    extra={
                        "computer": type(computer).__name__,
                        "missing_inputs": missing_inputs,
                    },
                )
            failed = _validate_computer(plugin=plugin, request=request, computer=computer)
            if failed is not None:
                return failed
            try:
                result = await computer.compute(request)
            except ValueError as exc:
                return _failed_result(
                    request,
                    "feature_computer_failed",
                    detail=str(exc),
                    feature_family=feature_family,
                )
            if result.feature_set_version != request.feature_set_version:
                return _failed_result(
                    request,
                    "feature_set_version_mismatch",
                    detail=(f"{result.feature_set_version!r} != {request.feature_set_version!r}"),
                    feature_family=feature_family,
                )
            if not result.passed:
                return result
            vectors.extend(result.vectors)
        return FeatureResult(
            feature_set_version=request.feature_set_version,
            vectors=tuple(vectors),
            diagnostics={
                "feature_family": feature_family,
                "feature_set_version": request.feature_set_version,
                "computer_count": len(computers),
            },
            passed=True,
        )


def _validate_computer(
    *,
    plugin: FeatureFamilyPlugin,
    request: FeatureRequest,
    computer: FeatureComputer,
) -> FeatureResult | None:
    feature_family = str(computer.feature_family)
    feature_set_version = str(computer.feature_set_version)
    output_features = tuple(str(name) for name in computer.output_features)
    schema_hash = str(computer.schema_hash)
    if feature_family != plugin.name:
        return _failed_result(
            request,
            "feature_family_mismatch",
            detail=f"{feature_family!r} != {plugin.name!r}",
            feature_family=plugin.name,
        )
    if feature_set_version != plugin.feature_set_version:
        return _failed_result(
            request,
            "feature_set_version_mismatch",
            detail=f"{feature_set_version!r} != {plugin.feature_set_version!r}",
            feature_family=plugin.name,
        )
    expected_schema_hash = ordered_feature_schema_hash(output_features)
    if schema_hash != expected_schema_hash:
        return _failed_result(
            request,
            "feature_schema_hash_mismatch",
            detail=f"{schema_hash!r} != {expected_schema_hash!r}",
            feature_family=plugin.name,
        )
    return None


def _missing_required_inputs(
    required_inputs: tuple[str, ...],
    request: FeatureRequest,
) -> tuple[str, ...]:
    available = set(request.input_context.available_inputs)
    return tuple(
        sorted(str(input_name) for input_name in required_inputs if input_name not in available)
    )


def _failed_result(
    request: FeatureRequest,
    blocker: str,
    *,
    detail: str,
    feature_family: str,
    extra: dict[str, object] | None = None,
) -> FeatureResult:
    diagnostics: dict[str, object] = {
        "feature_family": feature_family,
        "feature_set_version": request.feature_set_version,
        "blockers": (blocker,),
        "detail": detail,
    }
    if extra:
        diagnostics.update(extra)
    return FeatureResult(
        feature_set_version=request.feature_set_version,
        vectors=(),
        diagnostics=diagnostics,
        passed=False,
    )


__all__ = ["FeatureFamilyRegistry", "feature_plugin_key"]
