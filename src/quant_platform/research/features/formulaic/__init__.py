"""``formulaic-alpha-v1`` feature family.

This package implements a WorldQuant-style expression engine + a
starter library of hand-picked alphas. Each alpha is a programmatically-
constructed :class:`~.ast.Expression`; the evaluator walks the AST
against a :class:`~.panel.MarketPanel` to produce a feature column.
:class:`FeatureSpec` metadata (``required_inputs``, ``lookback_days``,
``point_in_time``) is derived automatically from the AST — there is no
sidecar manifest to hand-maintain.

How to add an alpha
-------------------

1. Build the AST using the operator builders in :mod:`.operators`
   (``rank``, ``delta``, ``ts_corr``, etc.) and :class:`.ast.Var` for
   input columns.
2. Append a :class:`~.library.FormulaicAlpha` entry to
   :data:`~.library.LIBRARY` with a stable ``name``, the expression,
   and a one-paragraph ``description``.
3. Add a smoke test under
   ``tests/unit/research_service/features/formulaic/test_library.py``.
4. Walk-forward evidence decides admission; nothing here declares an
   a-priori direction.

How to add an operator
----------------------

1. Implement the compute function (``_compute_foo(panel, series_args,
   *scalar_args)``) in :mod:`.operators`.
2. Register it in :data:`~.operators.OPERATORS` with the right axis
   (``time_series`` / ``cross_sectional`` / ``element_wise``).
3. Expose a public builder (``foo(x, window)``) that constructs the
   :class:`~.ast.OpCall` with the correct ``window_lookback`` so
   lookback derivation stays automatic.
4. Add per-operator unit tests under
   ``tests/unit/research_service/features/formulaic/test_operators.py``.

Auto-promoted alphas
--------------------

Alphas discovered by the mining loop (:mod:`.mining`) and promoted
via ``scripts/promote_alphas.py`` land in
:mod:`.auto_library` as JSONL rows. This package's import-time
``EFFECTIVE_LIBRARY`` concatenates the curated starter library with
the auto-promoted set, so :data:`FEATURE_SPECS` and
:func:`compute_formulaic_features` see both uniformly. Set
``QUANT_DISABLE_AUTO_PROMOTED_LIBRARY=1`` to force the family back
to "curated only" (used by deterministic CI tests).
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.formulaic.config import (
    DEFAULT_CONFIG,
    FEATURE_SET_VERSION,
    OPERATOR_SET_VERSION,
    FormulaicConfig,
)
from quant_platform.research.features.formulaic.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    EFFECTIVE_LIBRARY,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_formulaic_features,
)
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="formulaic",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side-effect: importing this package registers the manifest into the
# process-global registry.
register_family(MANIFEST)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "EFFECTIVE_LIBRARY",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "OPERATOR_SET_VERSION",
    "REQUIRED_INPUT_COLUMNS",
    "FormulaicConfig",
    "compute_formulaic_features",
]
