"""Configuration for the ``regime-v1`` feature family.

The regime family produces three classes of columns at the
``(instrument_id, date)`` grain:

1. **Date-keyed regime label/indicators** (constant across instruments
   on the same date). Surfaced for diagnostics; they have IC = 0
   cross-sectionally because every instrument on date ``D`` shares the
   same value — the IC-weighted ranker will weight them at 0.
2. **Date-keyed regime statistics** — ``trend_z``, ``realized_vol``,
   ``breadth`` from the upstream :class:`MarketStats`. Same shape as
   the indicators (one value per date, broadcast across instruments).
3. **Regime × base-feature interactions** — the columns that actually
   vary cross-sectionally and have non-zero IC. For each curated
   :class:`RegimeInteractionSpec`, emit either ``base × is_regime``
   (positive orientation) or ``-base × is_regime`` (negated /
   "anti-" orientation). The negated form is critical: the IC ranker
   runs with ``non_negative=True``, so it clips negative in-sample IC
   to zero. A column expected to invert sign in a regime (e.g.,
   momentum reverses in risk-off) can only contribute if it is
   emitted with the *expected* orientation already baked in. See
   ADR-005 "Hardening (post-review)" for the rationale.

The curation is deliberately conservative for the MVP — a small set
of high-conviction base features × the 3 most distinct regimes.
Expanding the cross-product is a follow-up tuning sweep, not a
default-add: a 4-regime × 36-base cross product is 144 columns, which
would risk overfitting against ~63 OOS days per regime per fold.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

FEATURE_SET_VERSION: str = "regime-v1"


@dataclass(frozen=True)
class RegimeInteractionSpec:
    """One ``(base_feature, regime_label, sign)`` interaction.

    ``negate=False`` emits ``f"{base}__x__{regime}"`` carrying the
    base-feature value when the indicator is 1 (and 0 otherwise).
    ``negate=True`` emits ``f"anti_{base}__x__{regime}"`` carrying the
    *negated* base-feature value when the indicator is 1.

    The split exists because the IC-weighted ranker runs with
    ``non_negative=True``: positive-IC features survive, negative-IC
    features are clipped to weight 0. A base feature whose effect
    *flips sign* inside a regime cannot survive without an explicit
    anti-variant — emitting both lets the ranker pick the one with
    positive in-sample IC for that regime.
    """

    base_feature: str
    regime_label: str
    negate: bool = False

    @property
    def column_name(self) -> str:
        prefix = "anti_" if self.negate else ""
        return f"{prefix}{self.base_feature}__x__{self.regime_label}"


#: Curated interaction set. Each entry is a
#: :class:`RegimeInteractionSpec` — see ADR-005 for the curation
#: rationale.
#:
#: Orientation choices reflect academic / empirical priors:
#:
#: * **Trend × RISK_ON** (positive orientation): trend-following works
#:   when the market is risk-on (momentum continuation premium).
#: * **Trend × RISK_OFF** (NEGATED orientation): trend reverses in
#:   stress — a textbook prior. Emitting the anti-variant lets the
#:   non-negative ranker express the sign flip.
#: * **Reversal × RISK_OFF/CRISIS** (positive): short-horizon mean
#:   reversion strengthens in stress.
#: * **Vol × RISK_ON/CRISIS** (positive): low-vol anomaly varies
#:   sharply across regimes; positive orientation is "high recent vol
#:   predicts low forward return" inside the regime.
#:
#: A base feature that doesn't exist in the upstream panel is silently
#: skipped — never raise on a missing input, because users may run
#: with feature subsets (e.g., ``--arms A`` uses only PV).
DEFAULT_INTERACTIONS: tuple[RegimeInteractionSpec, ...] = (
    # Trend × risk-on (positive)
    RegimeInteractionSpec("mom_12_1", "risk_on", negate=False),
    RegimeInteractionSpec("mom_6_1", "risk_on", negate=False),
    RegimeInteractionSpec("ret_252d", "risk_on", negate=False),
    # Trend × risk-off (NEGATED — sign-flip prior)
    RegimeInteractionSpec("mom_12_1", "risk_off", negate=True),
    RegimeInteractionSpec("mom_6_1", "risk_off", negate=True),
    RegimeInteractionSpec("ret_252d", "risk_off", negate=True),
    # Reversal × stress regimes (positive)
    RegimeInteractionSpec("reversal_5d", "risk_off", negate=False),
    RegimeInteractionSpec("reversal_5d", "crisis", negate=False),
    # Vol × extreme regimes (positive)
    RegimeInteractionSpec("vol_60d", "risk_on", negate=False),
    RegimeInteractionSpec("vol_60d", "crisis", negate=False),
    RegimeInteractionSpec("vol_21d", "risk_off", negate=False),
    RegimeInteractionSpec("vol_21d", "crisis", negate=False),
)


#: Regime indicators emitted as standalone columns. IC = 0 by
#: construction (constant per date); kept for audit/diagnostics so a
#: reviewer can see which regime applied on each date.
REGIME_INDICATOR_LABELS: tuple[str, ...] = (
    "risk_on",
    "risk_off",
    "transition",
    "crisis",
)


#: Date-keyed regime statistics emitted as panel columns. Same
#: cross-sectional-constant property as indicators, but kept because
#: a future Shape C (per-regime models) may use them as routing keys
#: or threshold inputs.
REGIME_STAT_COLUMNS: tuple[str, ...] = (
    "regime_trend_z",
    "regime_realized_vol",
    "regime_breadth",
)


#: Stable string identifiers describing how the research family
#: derives the regime classifier's inputs. Pinned into evidence /
#: run manifest so a future change to the proxy strategy invalidates
#: old Arm-H evidence rather than silently producing different labels
#: under the same family version.
INDEX_PROXY_ID: str = "universe_mean_close"
BREADTH_SOURCE_ID: str = "per_instrument_close_vs_50d_sma"


@dataclass(frozen=True)
class RegimeFeatureConfig(BaseFamilyConfig):
    """Frozen config for the regime feature family.

    The detector thresholds come from ``core.regime.DEFAULT_REGIME_THRESHOLDS``
    and are NOT re-tuned here — they are calibrated for live use, and
    research/live divergence would break the audit story. If a future
    PR needs different thresholds for research, the right move is to
    bump ``FEATURE_SET_VERSION`` and pin the thresholds in the
    artifact / evidence.
    """

    version: str = FEATURE_SET_VERSION

    #: Trailing window for the index-proxy "trend" z-score. Matches
    #: the detector's default 200-day SMA.
    trend_window: int = 200

    #: Trailing window for realised volatility. Matches the detector's
    #: 21-day annualised vol.
    vol_window: int = 21

    #: Trailing window for the breadth fraction. Matches the
    #: detector's 50-day SMA reference.
    breadth_window: int = 50


DEFAULT_CONFIG: RegimeFeatureConfig = RegimeFeatureConfig()


__all__ = [
    "BREADTH_SOURCE_ID",
    "DEFAULT_CONFIG",
    "DEFAULT_INTERACTIONS",
    "FEATURE_SET_VERSION",
    "INDEX_PROXY_ID",
    "REGIME_INDICATOR_LABELS",
    "REGIME_STAT_COLUMNS",
    "RegimeFeatureConfig",
    "RegimeInteractionSpec",
]
