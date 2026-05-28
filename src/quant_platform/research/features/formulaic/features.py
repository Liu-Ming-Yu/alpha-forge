"""Top-level compute + FeatureSpec derivation for ``formulaic-alpha-v1``.

The library in :mod:`.library` ships a tuple of :class:`FormulaicAlpha`
entries (name + expression + description). This module:

1. **Derives a `FeatureSpec` per library entry** from the AST. The
   spec's ``required_inputs`` and ``lookback_days`` come from
   :meth:`Expression.required_inputs` / :meth:`Expression.lookback_days`
   — no hand-maintained sidecar metadata.

2. **Evaluates each expression** against a built
   :class:`~.panel.MarketPanel` and assembles the
   :class:`FeatureFrame`. A shared :class:`ExpressionCache` is passed
   across alphas so sub-expressions that recur across the library
   (``rank(close)``, ``rank(volume)``, …) evaluate once per compute
   pass.

3. **Exports the standard family surface** (`FEATURE_SPECS`,
   `FEATURE_NAMES`, `DEFAULT_TRAINING_FEATURE_NAMES`) so the package
   ``__init__.py`` can build a :class:`FamilyManifest` the same way
   every other family does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.formulaic.auto_library import (
    load_promoted_library,
)
from quant_platform.research.features.formulaic.config import (
    DEFAULT_CONFIG,
    FormulaicConfig,
)
from quant_platform.research.features.formulaic.evaluator import (
    ExpressionCache,
    evaluate_expression,
)
from quant_platform.research.features.formulaic.library import LIBRARY, FormulaicAlpha
from quant_platform.research.features.formulaic.panel import (
    REQUIRED_INPUT_COLUMNS,
    build_market_panel,
)
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


def _spec_from_alpha(alpha: FormulaicAlpha, version: str) -> FeatureSpec:
    return FeatureSpec(
        name=alpha.name,
        family="formulaic",
        description=alpha.description,
        expected_direction=alpha.expected_direction,
        required_inputs=tuple(sorted(alpha.expression.required_inputs())),
        point_in_time=alpha.expression.point_in_time(),
        lookback_days=alpha.expression.lookback_days(),
        version=version,
        larger_is_better=alpha.larger_is_better,
    )


def _build_specs(version: str) -> tuple[FeatureSpec, ...]:
    return tuple(_spec_from_alpha(alpha, version) for alpha in EFFECTIVE_LIBRARY)


#: Curated starter library + any alphas auto-promoted from prior
#: mining runs (loaded from the JSONL registry at module import).
#: The two lists are concatenated in (curated-first, auto-second)
#: order so a name collision between curated and auto resolves in
#: favour of the curated definition.
def _build_effective_library() -> tuple[FormulaicAlpha, ...]:
    promoted = load_promoted_library()
    curated_names = {alpha.name for alpha in LIBRARY}
    deduped_promoted = tuple(alpha for alpha in promoted if alpha.name not in curated_names)
    return tuple(LIBRARY) + deduped_promoted


EFFECTIVE_LIBRARY: tuple[FormulaicAlpha, ...] = _build_effective_library()

FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(DEFAULT_CONFIG.version)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)
_SPEC_BY_NAME: Mapping[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}


def _specs_for_config(config: FormulaicConfig) -> tuple[FeatureSpec, ...]:
    """Return specs tagged with ``config.version``.

    Mirrors the helper of the same name in the price-volume and
    fundamentals families: production config returns the cached
    :data:`FEATURE_SPECS`, custom-version configs rebuild specs with
    the requested version pinned.
    """
    if config.version == DEFAULT_CONFIG.version:
        return FEATURE_SPECS
    return _build_specs(config.version)


def compute_formulaic_features(
    bars: pd.DataFrame,
    *,
    config: FormulaicConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``formulaic-alpha-v1`` panel.

    Parameters
    ----------
    bars:
        Long-format OHLCV bar frame, same shape as
        :func:`~..price_volume.features.compute_price_volume_features`
        accepts. Adapted internally via
        :func:`~.panel.build_market_panel`, which derives ``returns``
        and ``dollar_volume`` if they are not already columns.
    config:
        :class:`FormulaicConfig`. Defaults to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    FeatureFrame
        Long-format frame keyed by ``(instrument_id, date)`` with one
        column per library alpha. Each spec carries the library's
        version pin and the AST-derived ``required_inputs`` /
        ``lookback_days``.
    """
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}

    panel = build_market_panel(bars)
    cache = ExpressionCache()
    columns: dict[str, pd.Series] = {}
    for alpha in EFFECTIVE_LIBRARY:
        series = evaluate_expression(panel, alpha.expression, cache=cache)
        # ``±inf`` can still escape via inner divisions when the safe-div
        # guard doesn't catch them (e.g. ts_corr on a constant pair).
        # Sweep at the family boundary so every produced feature is a
        # clean float.
        columns[alpha.name] = series.replace([np.inf, -np.inf], np.nan).astype(float)

    output = panel.frame[list(DEFAULT_KEY_COLUMNS)].copy()
    for name in feature_names:
        output[name] = columns[name].to_numpy()

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
    "EFFECTIVE_LIBRARY",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_formulaic_features",
]
