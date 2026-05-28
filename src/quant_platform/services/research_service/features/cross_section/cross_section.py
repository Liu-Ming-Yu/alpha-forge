"""Cross-sectional feature computation pipeline.

Converts raw per-instrument close price series into normalized feature vectors
that can be fed directly to ``LinearWeightSignalModel`` and
``VolTargetedPortfolioConstructor``.

Pipeline for each rebalance cycle:
1. Compute raw factor values across the universe.
2. Winsorize each cross-section to remove extreme outliers.
3. Rank-normalize factors to [-1, 1].
4. Transpose factor-keyed data into per-instrument feature dicts.
5. Extract raw volatility forecasts for vol-targeted sizing.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.research_service.features.factors import (
    InsufficientDataError,
    realized_vol,
)

from .cross_section_factors import (
    STANDARD_FACTOR_SPECS,
    FactorSpec,
)
from .cross_section_normalization import (
    blend_factors,
    rank_normalize,
    winsorize,
    z_score_normalize,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

log = structlog.get_logger(__name__)

__all__ = [
    "FeatureBundle",
    "FactorSpec",
    "STANDARD_FACTOR_SPECS",
    "blend_factors",
    "build_feature_bundle",
    "rank_normalize",
    "winsorize",
    "z_score_normalize",
]


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBundle:
    """Output of ``build_feature_bundle``.

    Separates alpha signals (for ``LinearWeightSignalModel``) from volatility
    forecasts (for ``VolTargetedPortfolioConstructor``).

    Args:
        alpha_features: Per-instrument feature dicts.  Keys match
            ``LinearWeightSignalModel`` weight keys.  Values in [-1, 1]
            (rank-normalised).  Instruments with no computable factors
            are absent from this dict.
        vol_forecasts: Per-instrument annualised realised volatility (raw,
            not normalised).  Passed to ``set_vol_forecasts()`` before each
            portfolio construction cycle.  Instruments with insufficient history
            for the vol calculation are absent.
    """

    alpha_features: dict[uuid.UUID, dict[str, float]] = field(default_factory=dict)
    vol_forecasts: dict[uuid.UUID, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def build_feature_bundle(
    bar_data: Mapping[uuid.UUID, Sequence[float]],
    factor_specs: list[FactorSpec] | None = None,
    vol_window: int = 21,
) -> FeatureBundle:
    """Build normalised alpha feature vectors and raw vol forecasts.

    This is the main entry point for the daily feature generation step.

    Pipeline:
    1. For each factor spec, compute raw values across all instruments.
       Instruments with ``InsufficientDataError`` are skipped for that factor.
    2. Winsorise each factor's cross-section to clip outliers.
    3. Rank-normalise each factor to [-1, 1].
    4. Transpose from factor -> {instrument: value} to
       instrument -> {factor: value}.
    5. Instruments absent from all alpha factors receive no feature dict.
    6. Extract raw vol forecasts for ``VolTargetedPortfolioConstructor``.

    Args:
        bar_data: Mapping of instrument_id -> daily close prices (oldest first).
        factor_specs: Factor specifications to apply.  Defaults to
            ``STANDARD_FACTOR_SPECS``.
        vol_window: Look-back window for the vol forecast used in
            ``FeatureBundle.vol_forecasts``.  Matches the ``realized_vol_21d``
            factor by default.

    Returns:
        ``FeatureBundle`` with alpha features and vol forecasts.
        Instruments with insufficient data for all factors are excluded.
    """
    specs = factor_specs if factor_specs is not None else STANDARD_FACTOR_SPECS

    # Steps 1-3: compute, winsorize, rank-normalize each factor.
    # Structure: factor_name -> {instrument_id: normalised_score}
    normalized: dict[str, dict[uuid.UUID, float]] = {}

    for spec in specs:
        raw: dict[uuid.UUID, float] = {}
        for instr_id, closes in bar_data.items():
            try:
                raw[instr_id] = spec.compute(closes)
            except InsufficientDataError:
                pass
            except (ValueError, ZeroDivisionError):
                pass

        if not raw:
            continue

        winsorized = winsorize(raw, spec.winsorize_lower, spec.winsorize_upper)
        normalized[spec.name] = rank_normalize(winsorized)

    # Step 4: decide which instruments are retained in ``alpha_features``.
    #
    # The earlier "any alpha factor computed" rule let instruments with short
    # history slip through: a 10-bar series would fail ``momentum_1m``
    # (needs 22 bars) but succeed for ``short_term_reversal_5d`` (needs 6),
    # so the instrument ended up in ``alpha_features`` with neutral 0.0s for
    # every long-horizon factor.  That silently polluted the cross-section.
    #
    # Majority rule: an instrument is retained only if it was successfully
    # computed for at least half of the alpha factors that produced *any*
    # cross-section this cycle.  Factors that produced nothing for the whole
    # universe are excluded from the denominator so a mis-configured factor
    # cannot by itself cut the cohort to zero.
    alpha_feature_names = {spec.name for spec in specs if spec.is_alpha}
    computed_alpha_factors = [name for name in alpha_feature_names if normalized.get(name)]
    required_hits = (len(computed_alpha_factors) + 1) // 2  # ceil(n / 2)

    hit_counts: dict[uuid.UUID, int] = {}
    for factor_name in computed_alpha_factors:
        for instr_id in normalized[factor_name]:
            hit_counts[instr_id] = hit_counts.get(instr_id, 0) + 1

    retained_instruments = {
        instr_id for instr_id, hits in hit_counts.items() if hits >= required_hits
    }

    dropped = set(hit_counts) - retained_instruments
    if dropped:
        log.warning(
            "cross_section.instruments_dropped",
            count=len(dropped),
            required_hits=required_hits,
            sample=[str(i) for i in sorted(dropped, key=str)[:10]],
        )

    alpha_features: dict[uuid.UUID, dict[str, float]] = {}
    for instr_id in retained_instruments:
        feat: dict[str, float] = {}
        for factor_name in alpha_feature_names:
            factor_scores = normalized.get(factor_name)
            if factor_scores is None or instr_id not in factor_scores:
                # Omit the factor for this instrument rather than zero-filling.
                # ``LinearWeightSignalModel`` already discounts confidence for
                # missing features; injecting 0.0 silently masks coverage gaps
                # and pollutes the cross-section with neutral signals.
                continue
            feat[factor_name] = factor_scores[instr_id]
        alpha_features[instr_id] = feat

    # Step 5: extract raw (non-normalised) vol forecasts
    vol_forecasts: dict[uuid.UUID, float] = {}
    for instr_id, closes in bar_data.items():
        with contextlib.suppress(InsufficientDataError, ValueError, ZeroDivisionError):
            vol_forecasts[instr_id] = realized_vol(closes, window=vol_window, annualize=True)

    return FeatureBundle(
        alpha_features=alpha_features,
        vol_forecasts=vol_forecasts,
    )
