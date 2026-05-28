"""Research feature-family contracts and registries.

How this package is shaped
--------------------------

Two layers:

1. **Scaffold** — ``contracts.py`` (FeatureSpec, FeatureFrame, family
   enums), ``registry.py`` (process-global :class:`FeatureRegistry`),
   ``transforms.py`` (PIT-safe groupby/rolling helpers + shared tokens),
   ``neutralization.py`` (cross-sectional residualisers).
2. **Family packages** — one subpackage per feature family
   (``price_volume/``, ``fundamentals/``, …). Each family ships
   ``config.py``, ``features.py``, and ``__init__.py``; the
   ``__init__.py`` registers the family's specs into the global
   registry as a side effect of import.

Registration is automatic
-------------------------

Importing this package eagerly calls :func:`bootstrap_default_families`,
which imports every shipped family and (via their ``__init__.py``
side-effects) populates the global :class:`FeatureRegistry`. Callers
can therefore query ``get_global_registry()`` immediately after
``import quant_platform.research.features`` without juggling explicit
imports of each family. ``register_family`` is idempotent for
identical manifests, so re-importing is safe.

How to add a new feature family
-------------------------------

A new family ``foo`` is a copy-paste plus three edits:

1. Create ``src/quant_platform/research/features/foo/``.
2. Add ``foo/config.py``:
   * ``FEATURE_SET_VERSION: str`` — the family version string.
   * ``class FooConfig(BaseFamilyConfig)`` — family knobs.
   * ``DEFAULT_CONFIG: FooConfig`` — the canonical instance.
3. Add ``foo/features.py``:
   * Module-level ``REQUIRED_INPUT_COLUMNS``,
     ``FEATURE_SPECS``, ``FEATURE_NAMES``,
     ``DEFAULT_TRAINING_FEATURE_NAMES``.
   * ``compute_foo_features(panel, *, config) -> FeatureFrame`` that
     returns a :class:`FeatureFrame` with the family's
     ``key_columns``.
   * Every rolling / shift / groupby op must go through helpers in
     ``transforms.py`` so contributors don't reinvent
     ``sort=False, group_keys=False`` flags.
4. Add ``foo/__init__.py``:
   * Re-export the public surface.
   * Call ``register_many(FEATURE_SPECS)`` at module top-level.
5. Add tests under
   ``tests/unit/research_service/features/foo/``.

The :class:`FeatureSpec` docstring documents every field's contract.
Variant naming conventions (``low_*``, ``*_delta``, ``*_yoy``,
``*_ttm``) are described there too.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import (
    BaseFamilyConfig,
    FamilyManifest,
    FeatureDirection,
    FeatureFamily,
    FeatureFrame,
    FeatureSpec,
    SignalTimestamp,
)
from quant_platform.research.features.registry import (
    FeatureRegistry,
    get_global_registry,
    register,
    register_family,
    register_many,
)


def bootstrap_default_families() -> None:
    """Import every shipped family so it registers in the global registry.

    Family registration is a side effect of importing the family
    package (see each ``__init__.py``). This helper centralises that
    import chain so callers do not have to remember which families
    exist — useful in test fixtures, CLI startup, and the walk-forward
    bootstrap. Subsequent calls are no-ops because
    :meth:`FeatureRegistry.register_family` is idempotent for
    identical manifests.

    New families MUST be added to the import block below when they
    land; failing to do so leaves them unregistered until something
    else imports them.
    """
    import quant_platform.research.features.estimates  # noqa: F401
    import quant_platform.research.features.formulaic  # noqa: F401
    import quant_platform.research.features.fundamentals  # noqa: F401
    import quant_platform.research.features.learned  # noqa: F401
    import quant_platform.research.features.macro  # noqa: F401
    import quant_platform.research.features.microstructure  # noqa: F401
    import quant_platform.research.features.options  # noqa: F401
    import quant_platform.research.features.ownership  # noqa: F401
    import quant_platform.research.features.price_volume  # noqa: F401
    import quant_platform.research.features.regime  # noqa: F401
    import quant_platform.research.features.text  # noqa: F401


__all__ = [
    "BaseFamilyConfig",
    "FamilyManifest",
    "FeatureDirection",
    "FeatureFamily",
    "FeatureFrame",
    "FeatureRegistry",
    "FeatureSpec",
    "SignalTimestamp",
    "bootstrap_default_families",
    "get_global_registry",
    "register",
    "register_family",
    "register_many",
]


# Auto-populate the global registry with every shipped family on
# ``import quant_platform.research.features``. ``register_family`` is
# idempotent for identical manifests, so this is safe even when a
# downstream caller imports a family package directly afterwards.
# Kept at the bottom of the module so all the public exports above are
# defined before any family code runs.
bootstrap_default_families()
