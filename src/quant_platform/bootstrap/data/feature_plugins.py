"""Canonical assembly of the platform's executable feature-family plugins.

This module is the **single registration point** for runtime feature families.
It builds the typed :class:`~quant_platform.application.features.registry.FeatureFamilyRegistry`
from explicit :class:`~quant_platform.core.contracts.features.FeatureFamilyPlugin`
instances — replacing the string-keyed hardcoded dispatch dict.

Extensibility contract — adding a feature family/version:

1. Implement the feature math as a ``build_*_feature_bundle`` builder (or reuse
   an existing one).
2. Add a ``*_FEATURE_SET_VERSION`` constant and an ordered ``*_FEATURE_NAMES``
   tuple for the new output features.
3. Wrap the builder in a ``FeatureComputer`` — for bundle-backed families,
   ``BundleFeatureComputer`` already does this; otherwise implement the
   :class:`FeatureComputer` protocol directly.
4. Expose it through a :class:`FeatureFamilyPlugin` (``BundleFeatureFamilyPlugin``
   for the common case).
5. Append the plugin to :func:`build_research_feature_family_plugins` (in
   ``services/research_service/features/plugins.py``) — **nothing else changes**.
   The registry, fail-closed validation, the data-maintenance scheduler, and the
   discovery catalog all pick it up automatically.
6. Add a feature card and run the six-gate feature audit before promotion.

There is no dispatch dict to edit and no ``core/contracts`` change required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.features.registry import FeatureFamilyRegistry
from quant_platform.services.research_service.features.plugins import (
    build_research_feature_family_plugins,
)

if TYPE_CHECKING:
    from quant_platform.core.contracts import FeatureFamilyPlugin, FeatureRepository


def build_feature_family_plugins(
    feature_repo: FeatureRepository,
) -> tuple[FeatureFamilyPlugin, ...]:
    """Return every executable feature-family plugin wired into the platform.

    ``feature_repo`` is captured by families whose builders read prior governed
    feature vectors (text, catalyst, composite).
    """
    return build_research_feature_family_plugins(feature_repo)


def build_feature_registry(feature_repo: FeatureRepository) -> FeatureFamilyRegistry:
    """Return the canonical, fail-closed feature-family registry."""
    return FeatureFamilyRegistry.from_plugins(build_feature_family_plugins(feature_repo))


__all__ = ["build_feature_family_plugins", "build_feature_registry"]
