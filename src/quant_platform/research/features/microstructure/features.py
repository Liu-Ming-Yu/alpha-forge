"""``microstructure-v3`` feature factory.

Nineteen textbook microstructure proxies derived from daily OHLCV bars
at the ``(instrument_id, date)`` grain. The family is designed to be
**complementary to** ``price-volume-starter-v1`` — Amihud illiquidity,
dollar-volume z-score, plain high-low range, overnight gap, and the
open-to-close return already live there. This module ships features
that ``price-volume-starter-v1`` deliberately does not cover:

v1 features (10) — shipped first iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Range-based volatility estimators** (Parkinson, Garman-Klass,
  Rogers-Satchell). These use intra-day OHLC and are more efficient
  than close-to-close realised vol; Rogers-Satchell is additionally
  drift-independent.
* **Bid-ask spread proxies** computed from OHLC alone — Roll (1984)
  effective-spread estimator and Corwin-Schultz (2012) high-low
  spread estimator. Both stand in for the unobserved quoted spread
  on a daily-bar feed.
* **Intraday-position structure** — where in the day's range did
  the close land, on average. Proxy for buyer-vs-seller pressure
  without needing trade-direction signing.
* **Serial dependence** — return autocorrelation (bid-ask bounce
  signal) and volume autocorrelation (institutional-flow
  persistence).
* **Volume-return coupling** — rolling correlation of |return|
  and volume. High coupling = price moves on informative flow;
  low coupling = price moves on noise.
* **Range asymmetry** — rolling tilt between recent highs vs lows.

v2 additions (6)
~~~~~~~~~~~~~~~~

* **Yang-Zhang (2000) volatility** — drift-independent OHLC + overnight
  estimator. The gold-standard daily-OHLC vol; more efficient than
  Parkinson/Garman-Klass.
* **Bipower variation (Barndorff-Nielsen & Shephard 2004)** —
  jump-robust realised variance via Σ |rᵢ|·|rᵢ₋₁|.
* **Realized skew + kurtosis** — third and fourth standardised moments
  of daily returns. Distributional shape signals (crash risk, tail
  heaviness).
* **Lo-MacKinlay variance ratio (1988)** — VR(q) = Var(q-day returns)
  / (q × Var(1-day returns)). Random-walk null vs trend/reversal
  alternatives.
* **Range persistence** — rolling autocorrelation of the daily
  high-low range. Volatility clustering signature.

v3 additions (3) — improved jump-handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

v2's bipower variation is robust to **isolated** jumps but partially
contaminated by **clusters** (a jump immediately followed by a
same-magnitude mean reversion gives BPV a ``|r_t| · |r_{t-1}|``
product where both factors are large). v3 ships three estimators
that attack the jump problem from different angles:

* **Median Realized Variance** (Andersen-Dobrev-Schaumburg 2012) —
  uses ``med(|r_{t-2}|, |r_{t-1}|, |r_t|)²``. A single-day jump is
  the *maximum* of any 3-window, so the median ignores it — making
  MedRV **strictly better than BPV on isolated jumps**. Clusters
  (2 of 3 values large) still contaminate MedRV partially; for
  cluster regimes the right signal is ``realized_jump_intensity``
  below.
* **Tripower variation** (Barndorff-Nielsen & Shephard 2006) —
  three-return generalisation of bipower with the 2/3 exponent.
  Per-term contamination scales as ``J^(2/3) · ε^(4/3)`` instead
  of bipower's ``J · ε``, so isolated jumps leak less.
* **Realized jump intensity** (Andersen-Bollerslev-Diebold 2007) —
  ``(naive_RV − BPV) / naive_RV`` clipped to [0, 1]. The v2 BPV-vs-RV
  gap recast as a positive signal. Saturates near 1 on isolated
  jumps; partially elevated under clusters (the cluster regime is
  fundamentally harder on daily OHLCV — the true fix is intraday
  data, deferred to v4).

All nineteen features ship with ``expected_direction="unknown"`` and
``larger_is_better=False`` — they are evidence-gated by construction.
Promotion to a directional spec is a family-version bump, not an
in-place edit.

Tick / quote-level features (Kyle's λ, VPIN, true effective spread,
order-flow imbalance) defer to a future ``microstructure-v4`` once a
minute-bar or trade-tick feed is wired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.microstructure.config import (
    DEFAULT_CONFIG,
    MicrostructureConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    group_by_instrument,
    group_rolling_max,
    group_rolling_mean,
    group_rolling_min,
    group_rolling_std,
    group_shift,
    safe_div,
)

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


# ---------------------------------------------------------------------------
# v3 scale constants (computed at module load — no magic numbers)
# ---------------------------------------------------------------------------
#
# Both constants are checked against the closed-form derivation in the
# tests (test_med_rv_scale_constant_matches_ads_2012,
# test_tripower_scale_constant_matches_bns_2006) so a regression in
# either is caught at test time.


def _compute_med_rv_scale() -> float:
    """ADS 2012 MedRV scale: pi / (6 - 4*sqrt(3) + pi).

    Derived from E[ med(|Z_1|, |Z_2|, |Z_3|)^2 ] for iid Z ~ N(0, 1).
    Makes E[MedRV] equal the integrated variance under the
    diffusion-only null.
    """
    from math import pi, sqrt

    return pi / (6.0 - 4.0 * sqrt(3.0) + pi)


def _compute_tpv_scale() -> float:
    """BNS 2006 tripower-variation scale: mu_{2/3}^(-3), where
    mu_r = 2^(r/2) * Gamma((r+1)/2) / sqrt(pi) is the r-th absolute
    moment of N(0, 1). Specifically:

        mu_{2/3} = 2^(1/3) * Gamma(5/6) / sqrt(pi) ~ 0.8024
        mu_{2/3}^(-3) ~ 1.9358

    Makes E[TPV] equal the integrated variance under the
    diffusion-only null.
    """
    from math import gamma, pi, sqrt

    mu_two_thirds = (2 ** (1.0 / 3.0)) * gamma(5.0 / 6.0) / sqrt(pi)
    return 1.0 / (mu_two_thirds**3)


_MED_RV_SCALE: float = _compute_med_rv_scale()
_TPV_SCALE: float = _compute_tpv_scale()


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


def _build_specs(
    version: str,
    *,
    short_window: int,
    long_window: int,
    variance_ratio_stride: int = 5,
) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name=f"parkinson_vol_{short_window}d",
            family="microstructure",
            description=(
                f"Parkinson (1980) high-low range-based realised volatility "
                f"estimator over {short_window} trading days. "
                "sqrt( mean( ln(high/low)^2 ) / (4*ln(2)) ). "
                "More efficient than close-to-close vol because it uses "
                "the intra-day path; drift-dependent."
            ),
            expected_direction="unknown",
            required_inputs=("high", "low"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"garman_klass_vol_{short_window}d",
            family="microstructure",
            description=(
                f"Garman-Klass (1980) OHLC realised volatility estimator over "
                f"{short_window} trading days. "
                "sqrt( mean( 0.5*ln(high/low)^2 - (2*ln(2)-1)*ln(close/open)^2 ) ). "
                "Combines range + close-open; assumes zero drift."
            ),
            expected_direction="unknown",
            required_inputs=("open", "high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"rogers_satchell_vol_{short_window}d",
            family="microstructure",
            description=(
                f"Rogers-Satchell (1991) drift-independent OHLC realised "
                f"volatility estimator over {short_window} trading days. "
                "sqrt( mean( ln(high/close)*ln(high/open) + "
                "ln(low/close)*ln(low/open) ) ). "
                "Unbiased under non-zero drift, unlike Parkinson/Garman-Klass."
            ),
            expected_direction="unknown",
            required_inputs=("open", "high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"roll_spread_{long_window}d",
            family="microstructure",
            description=(
                f"Roll (1984) effective bid-ask spread estimator over "
                f"{long_window} trading days. 2*sqrt(-cov(r_t, r_{{t-1}})) "
                "when the return autocovariance is negative (bid-ask bounce "
                "signature); NaN otherwise. Daily-bar proxy for the unobserved "
                "quoted spread."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"corwin_schultz_spread_{short_window}d",
            family="microstructure",
            description=(
                f"Corwin-Schultz (2012) high-low bid-ask spread estimator over "
                f"{short_window} trading days. Derived from the variance of "
                "two-day vs one-day log-high-low ranges; tracks the quoted "
                "spread without quote data. Clipped at zero — negative raw "
                "estimates collapse to zero per the Corwin-Schultz convention."
            ),
            expected_direction="unknown",
            required_inputs=("high", "low"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"close_in_range_{short_window}d",
            family="microstructure",
            description=(
                f"Rolling mean over {short_window} trading days of "
                "(close - low) / (high - low) — where the close lands within "
                "the day's high-low range. Buyer-pressure proxy: values near "
                "1 = sessions consistently closing on highs; values near 0 = "
                "sessions consistently closing on lows."
            ),
            expected_direction="unknown",
            required_inputs=("high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"return_autocorr_{long_window}d",
            family="microstructure",
            description=(
                f"Rolling Pearson correlation between daily return and its "
                f"one-day lag, over {long_window} trading days. Negative "
                "values are the classic bid-ask bounce signature; persistent "
                "positive values indicate trend-following demand pressure."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"volume_autocorr_{long_window}d",
            family="microstructure",
            description=(
                f"Rolling Pearson correlation between daily volume and its "
                f"one-day lag, over {long_window} trading days. High = "
                "persistent institutional flow; low = idiosyncratic volume "
                "noise day-to-day."
            ),
            expected_direction="unknown",
            required_inputs=("volume",),
            point_in_time=True,
            lookback_days=long_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"volume_return_correlation_{short_window}d",
            family="microstructure",
            description=(
                f"Rolling Pearson correlation between |daily return| and "
                f"daily volume over {short_window} trading days. High = "
                "price moves with volume (informative flow); low = price "
                "moves on low volume (noise / liquidity-driven)."
            ),
            expected_direction="unknown",
            required_inputs=("close", "volume"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"high_low_asymmetry_{short_window}d",
            family="microstructure",
            description=(
                f"Range tilt over {short_window} trading days: "
                "(rolling_max(high) - close) / (close - rolling_min(low)). "
                "Values > 1 = recent highs further away than recent lows "
                "(upside resistance dominates); values < 1 = recent lows "
                "further away (downside support dominates). "
                "NaN when the close equals the rolling low (zero denominator)."
            ),
            expected_direction="unknown",
            required_inputs=("high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window,
            version=version,
            larger_is_better=False,
        ),
        # v2 additions ------------------------------------------------
        FeatureSpec(
            name=f"yang_zhang_vol_{short_window}d",
            family="microstructure",
            description=(
                f"Yang-Zhang (2000) drift-independent OHLC + overnight "
                f"volatility estimator over {short_window} trading days. "
                "Combines overnight variance (ln(open/prev_close)^2), the "
                "open-to-close variance, and the Rogers-Satchell intraday "
                "contribution via the k-weighting that minimises estimator "
                "variance. The gold-standard daily-OHLC vol estimator — "
                "more efficient than Parkinson/Garman-Klass and unbiased "
                "under non-zero drift like Rogers-Satchell."
            ),
            expected_direction="unknown",
            required_inputs=("open", "high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window + 1,  # needs prev_close for overnight
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"bipower_variation_{short_window}d",
            family="microstructure",
            description=(
                f"Barndorff-Nielsen-Shephard (2004) jump-robust realised "
                f"variance estimator over {short_window} trading days. "
                "(pi/2) * mean( |r_t| * |r_{t-1}| ). Unlike sum(r^2), "
                "this is consistent for the integrated variance under "
                "jump-diffusion price processes — large isolated jumps "
                "drop out because each adjacent product touches at most "
                "one jump term."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=short_window + 1,  # needs the lagged |r|
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"realized_skew_{long_window}d",
            family="microstructure",
            description=(
                f"Third standardised moment of daily log returns over "
                f"{long_window} trading days. Negative = downside-heavy "
                "distribution (crash risk premium); positive = upside-"
                "heavy. Computed with the bias-corrected ``pandas.Series."
                "skew`` (Fisher-Pearson)."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"realized_kurt_{long_window}d",
            family="microstructure",
            description=(
                f"Excess kurtosis (fourth standardised moment minus 3, "
                f"Fisher's definition) of daily log returns over "
                f"{long_window} trading days. Positive = fat tails; 0 = "
                "Gaussian. Computed with the bias-corrected "
                "``pandas.Series.kurt`` (which already returns excess "
                "kurtosis, not raw 4th moment — the spec name omits "
                "``excess_`` per finance convention)."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"variance_ratio_{variance_ratio_stride}_1_{long_window}d",
            family="microstructure",
            description=(
                f"Lo-MacKinlay (1988) variance ratio with stride "
                f"q={variance_ratio_stride}, computed over a {long_window}-"
                "trading-day window. VR(q) = Var(q-day log returns) / "
                "(q * Var(1-day log returns)). Under a random walk VR(q) "
                "= 1; VR < 1 = mean reversion (bid-ask bounce or short-"
                "term overreaction); VR > 1 = positive serial correlation "
                "(trend / momentum at the q-day horizon). "
                "**Implementation note:** uses *overlapping* q-stride "
                "returns over a rolling window, not the original paper's "
                "non-overlapping returns. The resulting VR is biased "
                "relative to the textbook statistic (the bias is "
                "window-constant), but cross-row deviations from 1.0 "
                "remain interpretable as serial-correlation regimes. "
                "NaN on rows where the rolling 1-day variance is "
                "non-positive (zero-variance instruments drop out)."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window + variance_ratio_stride,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"range_persistence_{short_window}d",
            family="microstructure",
            description=(
                f"Per-instrument rolling autocorrelation of the daily "
                f"high-low range (high - low) / close, over {short_window} "
                "trading days. High = persistent volatility regime "
                "(garch-like clustering); low/negative = noisy range that "
                "doesn't carry day-to-day."
            ),
            expected_direction="unknown",
            required_inputs=("high", "low", "close"),
            point_in_time=True,
            lookback_days=short_window + 1,
            version=version,
            larger_is_better=False,
        ),
        # v3 additions — jump-cluster-robust estimators ------------
        FeatureSpec(
            name=f"med_rv_{long_window}d",
            family="microstructure",
            description=(
                f"Median Realized Variance (Andersen-Dobrev-Schaumburg "
                f"2012) over {long_window} trading days. Daily contribution "
                "is the squared median of three adjacent absolute log "
                "returns: med(|r_{t-2}|, |r_{t-1}|, |r_t|)^2; rolling mean "
                "over the long window with the ADS efficiency-correction "
                "constant pi / (6 - 4*sqrt(3) + pi) ~ 1.0726. **Genuinely "
                "robust to single-day jump clusters**: a single jump is the "
                "maximum of any 3-window so the median ignores it. "
                "Recommended as the jump-robust integrated-variance "
                "estimator, complementing bipower_variation_*d which fails "
                "on jump-and-mean-revert clusters."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=long_window + 2,  # needs lag-2 absolute return
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"tripower_variation_{short_window}d",
            family="microstructure",
            description=(
                f"Tripower variation (Barndorff-Nielsen-Shephard 2006) over "
                f"{short_window} trading days. Daily contribution is "
                "|r_t|^(2/3) * |r_{t-1}|^(2/3) * |r_{t-2}|^(2/3); rolling "
                "mean over the short window scaled by the BNS constant "
                "mu_{2/3}^(-3) ~ 1.934. Three-return generalisation of "
                "bipower with sub-linear exponent — more robust to "
                "multi-day jump clusters because the contamination per "
                "term scales as J^(2/3) * eps^(4/3) instead of J*eps."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=short_window + 2,  # needs lag-2 absolute return
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"realized_jump_intensity_{short_window}d",
            family="microstructure",
            description=(
                f"Andersen-Bollerslev-Diebold (2007) jump-intensity proxy "
                f"over {short_window} trading days: "
                "max(0, (naive_RV - bipower_variation) / naive_RV), where "
                "naive_RV is the rolling mean of squared log returns. "
                "Lives in [0, 1]: 0 = no detectable jumps (BPV ~ RV); "
                "1 = all realised variance is jump-driven. Higher = more "
                "jumpy regime. The v2 BPV-vs-RV gap, recast as a positive "
                "signal."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=short_window + 1,
            version=version,
            larger_is_better=False,
        ),
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    short_window=DEFAULT_CONFIG.short_window,
    long_window=DEFAULT_CONFIG.long_window,
    variance_ratio_stride=DEFAULT_CONFIG.variance_ratio_stride,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _validate_inputs(bars: pd.DataFrame) -> None:
    """Reject bar frames that are missing required OHLCV columns."""
    missing = [name for name in REQUIRED_INPUT_COLUMNS if name not in bars.columns]
    if missing:
        raise ValueError(
            "compute_microstructure_features: bars missing required columns: "
            f"{missing!r}; got {list(bars.columns)!r}"
        )


def _specs_for_config(config: MicrostructureConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.short_window == DEFAULT_CONFIG.short_window
        and config.long_window == DEFAULT_CONFIG.long_window
        and config.variance_ratio_stride == DEFAULT_CONFIG.variance_ratio_stride
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        short_window=config.short_window,
        long_window=config.long_window,
        variance_ratio_stride=config.variance_ratio_stride,
    )


def _per_instrument_rolling_op(
    df: pd.DataFrame,
    *,
    window: int,
    op: Callable[[pd.DataFrame, int], pd.Series],
) -> pd.Series:
    """Apply a per-instrument rolling-window operation and stitch the
    results back into one Series aligned with ``df.index``.

    The shared core for :func:`_rolling_corr`, :func:`_rolling_cov`, and
    :func:`_rolling_higher_moment`. Three operations, one iteration
    pattern.

    Explicit per-instrument iteration is used instead of
    ``DataFrameGroupBy.apply`` because pandas' apply returns a
    transposed shape on single-group frames (a deprecated behaviour
    that still triggers a length-mismatch error in tests with one
    instrument). Iterating + writing into a pre-allocated Series is
    correct on any group count.

    Performance note: each iteration step does one ``Rolling`` pass
    over a per-instrument slice. On a ~300-instrument × multi-year
    panel this is acceptable (~1200 Rolling calls across all four
    consumers). If the universe grows past ~1000 names, profile this
    helper before scaling further; the per-call Python overhead
    starts to dominate.

    Parameters
    ----------
    df:
        Long-format frame keyed by ``instrument_id``.
    window:
        Rolling-window length, passed to ``op`` as its second argument.
    op:
        Callable ``(per_instrument_group, window) -> pd.Series``. The
        returned Series must be indexed compatibly with the input
        group (i.e. carry the group's row labels). The shared core
        writes each group's result back into the pre-allocated output
        by row label.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _instrument, group in df.groupby("instrument_id", sort=False):
        result = op(group, window)
        out.loc[group.index] = result.to_numpy()
    return out


def _rolling_corr(
    df: pd.DataFrame,
    *,
    series_a_column: str,
    series_b_column: str,
    window: int,
) -> pd.Series:
    """Per-instrument rolling Pearson correlation between two columns."""
    return _per_instrument_rolling_op(
        df,
        window=window,
        op=lambda group, w: (
            group[series_a_column].rolling(w, min_periods=w).corr(group[series_b_column])
        ),
    )


def _rolling_cov(
    df: pd.DataFrame,
    *,
    series_a_column: str,
    series_b_column: str,
    window: int,
) -> pd.Series:
    """Per-instrument rolling covariance — used by Roll's spread
    estimator (``cov(r_t, r_{t-1})``)."""
    return _per_instrument_rolling_op(
        df,
        window=window,
        op=lambda group, w: (
            group[series_a_column].rolling(w, min_periods=w).cov(group[series_b_column])
        ),
    )


def _rolling_higher_moment(
    df: pd.DataFrame,
    *,
    column: str,
    window: int,
    method: str,
) -> pd.Series:
    """Per-instrument rolling skew or kurt. ``method`` is ``"skew"`` or
    ``"kurt"``; both use the Fisher-Pearson bias-corrected definitions
    that ``pandas.Series.rolling`` ships."""
    if method not in ("skew", "kurt"):
        raise ValueError(f"_rolling_higher_moment: unsupported method {method!r}")

    def _op(group: pd.DataFrame, w: int) -> pd.Series:
        rolling = group[column].rolling(w, min_periods=w)
        return rolling.skew() if method == "skew" else rolling.kurt()

    return _per_instrument_rolling_op(df, window=window, op=_op)


def _enforce_version_window_consistency(config: MicrostructureConfig) -> None:
    """Refuse to compute when the caller has bumped a window without
    bumping the family version.

    Window values (``short_window``, ``long_window``,
    ``variance_ratio_stride``) appear in every feature column name —
    silently producing ``parkinson_vol_10d`` under
    ``version="microstructure-v2"`` would make two runs with the same
    manifest emit different catalogues. This guard makes the contract
    explicit: same manifest version ⇒ same windows.
    """
    if config.version != DEFAULT_CONFIG.version:
        # Caller has explicitly bumped the version; they can choose any
        # windows they want.
        return
    drifts: list[str] = []
    if config.short_window != DEFAULT_CONFIG.short_window:
        drifts.append(f"short_window={config.short_window} (default {DEFAULT_CONFIG.short_window})")
    if config.long_window != DEFAULT_CONFIG.long_window:
        drifts.append(f"long_window={config.long_window} (default {DEFAULT_CONFIG.long_window})")
    if config.variance_ratio_stride != DEFAULT_CONFIG.variance_ratio_stride:
        drifts.append(
            f"variance_ratio_stride={config.variance_ratio_stride} "
            f"(default {DEFAULT_CONFIG.variance_ratio_stride})"
        )
    if drifts:
        raise ValueError(
            "MicrostructureConfig: a window/stride was changed without bumping "
            "the family version. Doing so would silently produce a different "
            "feature catalogue under the same manifest version. Either keep "
            "all windows at their defaults, or set version != "
            f"{DEFAULT_CONFIG.version!r}. Drifted: {drifts!r}."
        )


def compute_microstructure_features(
    bars: pd.DataFrame,
    *,
    config: MicrostructureConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``microstructure-v2`` feature panel.

    Parameters
    ----------
    bars:
        Long-format DataFrame keyed by ``(instrument_id, date)`` with
        OHLCV columns. ``volume`` may be zero on stale rows; features
        that consume it tolerate that via :func:`safe_div`.
    config:
        :class:`MicrostructureConfig`.

    Returns
    -------
    FeatureFrame
        Sixteen microstructure features per ``(instrument_id, date)``.
    """
    _validate_inputs(bars)
    # Refuse to compute when the caller drifted a window without
    # bumping the family version — would produce a v2-stamped manifest
    # with v2-incompatible feature column names. See
    # ``_enforce_version_window_consistency`` for the full rule.
    _enforce_version_window_consistency(config)
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    short_w = config.short_window
    long_w = config.long_window
    vr_stride = config.variance_ratio_stride

    if bars.empty:
        empty = pd.DataFrame(
            {col: pd.Series(dtype=bars[col].dtype) for col in REQUIRED_INPUT_COLUMNS}
        )
        for name in feature_names:
            empty[name] = pd.Series(dtype=float)
        return FeatureFrame(
            frame=empty[["instrument_id", "date", *feature_names]].copy(),
            feature_names=feature_names,
            feature_specs=spec_by_name,
            coverage={name: 0 for name in feature_names},
            key_columns=DEFAULT_KEY_COLUMNS,
        )

    df = bars.loc[:, list(REQUIRED_INPUT_COLUMNS)].copy()
    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-day terms (no group-by needed — these are row-local).
    # ------------------------------------------------------------------
    # Guard against zero/negative inputs into log: replace non-positive
    # with NaN so log returns NaN rather than -inf / NaN-with-warnings.
    safe_high = df["high"].where(df["high"] > 0, np.nan)
    safe_low = df["low"].where(df["low"] > 0, np.nan)
    safe_open = df["open"].where(df["open"] > 0, np.nan)
    safe_close = df["close"].where(df["close"] > 0, np.nan)

    ln_hl = np.log(safe_high / safe_low)
    ln_co = np.log(safe_close / safe_open)
    ln_hc = np.log(safe_high / safe_close)
    ln_ho = np.log(safe_high / safe_open)
    ln_lc = np.log(safe_low / safe_close)
    ln_lo = np.log(safe_low / safe_open)

    df["_parkinson_daily"] = (ln_hl**2) / (4.0 * np.log(2.0))
    df["_gk_daily"] = 0.5 * (ln_hl**2) - (2.0 * np.log(2.0) - 1.0) * (ln_co**2)
    df["_rs_daily"] = ln_hc * ln_ho + ln_lc * ln_lo

    # Close-in-range per day: position in [0, 1]. NaN when high == low
    # (no range to normalise into).
    range_today = df["high"] - df["low"]
    df["_close_in_range_daily"] = safe_div(
        df["close"] - df["low"], range_today, require_positive_denom=True
    )

    # Daily log return for Roll's spread estimator + return autocorr.
    grouped = group_by_instrument(df)
    close_lag1 = group_shift(grouped["close"], 1)
    df["_log_ret"] = np.log(safe_div(df["close"], close_lag1, require_positive_denom=True))
    df["_abs_log_ret"] = df["_log_ret"].abs()

    # ------------------------------------------------------------------
    # Rolling features.
    # ------------------------------------------------------------------
    grp_pk = group_by_instrument(df)["_parkinson_daily"]
    grp_gk = group_by_instrument(df)["_gk_daily"]
    grp_rs = group_by_instrument(df)["_rs_daily"]
    grp_cir = group_by_instrument(df)["_close_in_range_daily"]

    # Range-based vols: sqrt of the rolling MEAN of the daily
    # contribution. Take sqrt only on the non-negative part (RS can
    # be slightly negative on outlier OHLC rows; clip before sqrt).
    pk_var = group_rolling_mean(grp_pk, short_w, policy="full")
    gk_var = group_rolling_mean(grp_gk, short_w, policy="full")
    rs_var = group_rolling_mean(grp_rs, short_w, policy="full")
    df[f"parkinson_vol_{short_w}d"] = np.sqrt(np.clip(pk_var, 0.0, None))
    df[f"garman_klass_vol_{short_w}d"] = np.sqrt(np.clip(gk_var, 0.0, None))
    df[f"rogers_satchell_vol_{short_w}d"] = np.sqrt(np.clip(rs_var, 0.0, None))

    # Close-in-range mean.
    df[f"close_in_range_{short_w}d"] = group_rolling_mean(grp_cir, short_w, policy="full")

    # Roll's spread estimator: 2*sqrt(-cov(r_t, r_{t-1})). Per-instrument
    # rolling covariance via :func:`_rolling_cov` (explicit iteration —
    # safer than ``groupby.apply`` for single-instrument frames).
    df["_log_ret_lag1"] = group_shift(group_by_instrument(df)["_log_ret"], 1)
    cov_lr = _rolling_cov(
        df, series_a_column="_log_ret", series_b_column="_log_ret_lag1", window=long_w
    )
    # Roll's estimator is real-valued only when cov < 0 (bid-ask bounce
    # signature). When cov >= 0 we emit NaN per the original paper —
    # the estimator is undefined there. Suppress numpy's sqrt warning
    # on the NaN branch since we mask it out anyway.
    with np.errstate(invalid="ignore"):
        df[f"roll_spread_{long_w}d"] = np.where(cov_lr < 0, 2.0 * np.sqrt(-cov_lr), np.nan)

    # Corwin-Schultz spread estimator from two-day vs one-day high-low
    # log-ranges. Two-day range uses pairwise rolling max/min via
    # group_rolling_max / group_rolling_min on (high, low). Daily
    # contribution is then averaged over short_window.
    grp_h2 = group_rolling_max(group_by_instrument(df)["high"], 2, policy="full")
    grp_l2 = group_rolling_min(group_by_instrument(df)["low"], 2, policy="full")
    beta_today = ln_hl**2
    grp_beta = group_by_instrument(df.assign(_beta=beta_today))["_beta"]
    beta = group_rolling_mean(grp_beta, 2, policy="full") * 2.0
    gamma = np.log(grp_h2 / grp_l2.where(grp_l2 > 0, np.nan)) ** 2
    # Per Corwin-Schultz: alpha = (sqrt(2*beta) - sqrt(beta)) /
    # (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2))).
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(
        np.clip(gamma / denom, 0.0, None)
    )
    spread_daily = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # Negative raw spreads collapse to zero per the Corwin-Schultz
    # convention (a negative estimate has no microstructure interpretation).
    spread_daily = np.where(spread_daily < 0, 0.0, spread_daily)
    df["_cs_spread_daily"] = spread_daily
    df[f"corwin_schultz_spread_{short_w}d"] = group_rolling_mean(
        group_by_instrument(df)["_cs_spread_daily"], short_w, policy="full"
    )

    # Return + volume autocorrelations.
    df[f"return_autocorr_{long_w}d"] = _rolling_corr(
        df, series_a_column="_log_ret", series_b_column="_log_ret_lag1", window=long_w
    )
    df["_volume_lag1"] = group_shift(group_by_instrument(df)["volume"], 1)
    df[f"volume_autocorr_{long_w}d"] = _rolling_corr(
        df, series_a_column="volume", series_b_column="_volume_lag1", window=long_w
    )

    # Volume-return correlation: corr(|r|, volume) over short_window.
    df[f"volume_return_correlation_{short_w}d"] = _rolling_corr(
        df, series_a_column="_abs_log_ret", series_b_column="volume", window=short_w
    )

    # High-low asymmetry: (rolling_max(high) - close) / (close - rolling_min(low)).
    rolling_max_high = group_rolling_max(group_by_instrument(df)["high"], short_w, policy="full")
    rolling_min_low = group_rolling_min(group_by_instrument(df)["low"], short_w, policy="full")
    upside = rolling_max_high - df["close"]
    downside = df["close"] - rolling_min_low
    df[f"high_low_asymmetry_{short_w}d"] = safe_div(upside, downside, require_positive_denom=True)

    # ------------------------------------------------------------------
    # v2 additions
    # ------------------------------------------------------------------

    # Yang-Zhang (2000) drift-independent OHLC+overnight volatility.
    # sigma^2_yz = sigma^2_overnight + k * sigma^2_open_to_close + (1-k) * sigma^2_rs
    # where k = 0.34 / (1.34 + (N+1)/(N-1)) and N = short_window.
    #
    # overnight_log_ret_t = ln(open_t / close_{t-1})
    # open_to_close_log_ret_t = ln(close_t / open_t)  (already available as ln_co)
    df["_overnight_log_ret"] = np.log(safe_div(df["open"], close_lag1, require_positive_denom=True))
    df["_oc_log_ret"] = ln_co  # alias for clarity

    grp_overnight = group_by_instrument(df)["_overnight_log_ret"]
    grp_oc = group_by_instrument(df)["_oc_log_ret"]
    grp_rs_var = group_by_instrument(df)["_rs_daily"]

    # Variance of overnight + open-to-close log returns over short_w.
    # Use the bias-corrected sample variance to match Yang-Zhang's
    # original derivation; group_rolling_std defaults to ddof=1.
    # ``policy="full"`` is REQUIRED: the YZ k-weight below assumes the
    # rolling window saw exactly ``short_w`` observations. A
    # ``policy="partial"`` would let early rows enter the variance
    # estimator with fewer samples than ``short_w``, breaking the
    # k-weight's denominator scaling. The family doesn't currently
    # expose ``min_periods_policy`` on the config, but if it ever does,
    # the YZ path must stay pinned to "full".
    overnight_var = group_rolling_std(grp_overnight, short_w, policy="full") ** 2
    oc_var = group_rolling_std(grp_oc, short_w, policy="full") ** 2
    rs_var_yz = group_rolling_mean(grp_rs_var, short_w, policy="full")

    # YZ k-weight: 0.34 / (1.34 + (N+1)/(N-1)). The denominator
    # divides by (N-1), so N == 1 would blow up. Protected here by
    # ``MicrostructureConfig.short_window >= 5`` (config validator).
    # The explicit guard documents the coupling and would catch a
    # future config relaxation that lowered the floor below 2.
    n_yz = short_w
    if n_yz < 2:
        raise ValueError(
            f"Yang-Zhang k-weight requires short_window >= 2 (denominator "
            f"divides by short_window - 1); got short_window={n_yz}"
        )
    k_yz = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1))
    yz_var = overnight_var + k_yz * oc_var + (1.0 - k_yz) * rs_var_yz
    df[f"yang_zhang_vol_{short_w}d"] = np.sqrt(np.clip(yz_var, 0.0, None))

    # Bipower variation (Barndorff-Nielsen & Shephard 2004): jump-robust
    # realised variance. BPV_t = (pi/2) * sum_{i=2..N} |r_i| * |r_{i-1}|
    # over a rolling window of length N. Implementing as a rolling mean
    # of |r_t| * |r_{t-1}| * (pi/2) is equivalent up to the leading
    # constant; what matters is the relative magnitude.
    df["_abs_log_ret_lag1"] = group_shift(group_by_instrument(df)["_abs_log_ret"], 1)
    df["_bipower_product_daily"] = (np.pi / 2.0) * df["_abs_log_ret"] * df["_abs_log_ret_lag1"]
    df[f"bipower_variation_{short_w}d"] = group_rolling_mean(
        group_by_instrument(df)["_bipower_product_daily"], short_w, policy="full"
    )

    # Realized skew + kurt over long_w days. Routes through the shared
    # ``_rolling_higher_moment`` helper which wraps the bias-corrected
    # Fisher-Pearson definitions that ``pandas.Series.rolling`` ships.
    df[f"realized_skew_{long_w}d"] = _rolling_higher_moment(
        df, column="_log_ret", window=long_w, method="skew"
    )
    df[f"realized_kurt_{long_w}d"] = _rolling_higher_moment(
        df, column="_log_ret", window=long_w, method="kurt"
    )

    # Lo-MacKinlay variance ratio VR(q) with q = vr_stride. Compute over
    # a long_w-day window. The conventional formulation uses
    # non-overlapping q-day returns; for a rolling implementation we
    # use the cleaner overlapping-q-stride form, which is biased but
    # comparable across rows because the bias is window-constant
    # (the spec description documents this caveat for consumers).
    # q-day log return: ln(close_t / close_{t-q}).
    close_lag_q = group_shift(group_by_instrument(df)["close"], vr_stride)
    df["_q_stride_log_ret"] = np.log(
        safe_div(df["close"], close_lag_q, require_positive_denom=True)
    )

    var_1d = group_rolling_std(group_by_instrument(df)["_log_ret"], long_w, policy="full") ** 2
    var_q = (
        group_rolling_std(group_by_instrument(df)["_q_stride_log_ret"], long_w, policy="full") ** 2
    )
    df[f"variance_ratio_{vr_stride}_1_{long_w}d"] = safe_div(
        var_q, vr_stride * var_1d, require_positive_denom=True
    )

    # Range persistence: per-instrument autocorrelation of the daily
    # high-low range over short_w days.
    df["_range_norm_daily"] = safe_div(
        df["high"] - df["low"], df["close"], require_positive_denom=True
    )
    df["_range_norm_lag1"] = group_shift(group_by_instrument(df)["_range_norm_daily"], 1)
    df[f"range_persistence_{short_w}d"] = _rolling_corr(
        df, series_a_column="_range_norm_daily", series_b_column="_range_norm_lag1", window=short_w
    )

    # ------------------------------------------------------------------
    # v3 additions: jump-cluster-robust estimators
    # ------------------------------------------------------------------

    # Pre-compute lag-2 absolute log return; med_rv and tripower both need it.
    df["_abs_log_ret_lag2"] = group_shift(group_by_instrument(df)["_abs_log_ret"], 2)

    # Median Realized Variance (Andersen-Dobrev-Schaumburg 2012).
    # Daily contribution: med(|r_{t-2}|, |r_{t-1}|, |r_t|)^2.
    # The 3-row row-wise median is computed with pandas' DataFrame.median(axis=1),
    # which gracefully handles the per-instrument warm-up NaNs (median of
    # a row with NaN(s) just drops them and medians the rest — so the
    # warm-up rows return real (but biased low) values, which the long_w
    # rolling mean then absorbs with min_periods=long_w anyway).
    df["_med_abs_3"] = df[["_abs_log_ret_lag2", "_abs_log_ret_lag1", "_abs_log_ret"]].median(axis=1)
    df["_med_rv_daily"] = df["_med_abs_3"] ** 2
    # ADS 2012 efficiency-correction constant: pi / (6 - 4*sqrt(3) + pi)
    # ~ 1.4194. Derived from E[ med(|Z_1|, |Z_2|, |Z_3|)^2 ] for iid
    # Z ~ N(0, 1) — the normalisation that makes E[MedRV] equal the
    # integrated variance under the diffusion-only null. The constant
    # is checked against the closed-form in
    # test_med_rv_scale_constant_matches_ads_2012.
    df[f"med_rv_{long_w}d"] = _MED_RV_SCALE * group_rolling_mean(
        group_by_instrument(df)["_med_rv_daily"], long_w, policy="full"
    )

    # Tripower variation (Barndorff-Nielsen-Shephard 2006).
    # Daily contribution: |r_t|^(2/3) * |r_{t-1}|^(2/3) * |r_{t-2}|^(2/3).
    # Rolling mean scaled by mu_{2/3}^(-3) ~ 1.936, where
    # mu_r = E[|Z|^r] = 2^(r/2) * Gamma((r+1)/2) / sqrt(pi) for
    # Z ~ N(0, 1) — the normalisation that makes TPV a consistent
    # estimator of the integrated variance under the diffusion-only
    # null. The constant is computed at module load time from
    # ``math.gamma`` (no magic number to maintain) and checked
    # against the closed-form in
    # test_tripower_scale_constant_matches_bns_2006.
    df["_tripower_product_daily"] = (
        (df["_abs_log_ret"] ** (2.0 / 3.0))
        * (df["_abs_log_ret_lag1"] ** (2.0 / 3.0))
        * (df["_abs_log_ret_lag2"] ** (2.0 / 3.0))
    )
    df[f"tripower_variation_{short_w}d"] = _TPV_SCALE * group_rolling_mean(
        group_by_instrument(df)["_tripower_product_daily"], short_w, policy="full"
    )

    # Realized jump intensity (Andersen-Bollerslev-Diebold 2007).
    # naive_RV = rolling mean of r_t^2 over short_w (matches the BPV
    # window so the comparison is apples-to-apples).
    # jump_intensity = clip( (RV - BPV) / RV, 0, 1 ).
    # The clipping below zero is because numerical noise can give
    # negative gaps on very-smooth-return windows; we never report
    # "negative jump intensity" — that's not interpretable.
    df["_r_squared_daily"] = df["_log_ret"] ** 2
    rv_mean = group_rolling_mean(
        group_by_instrument(df)["_r_squared_daily"], short_w, policy="full"
    )
    bpv_for_intensity = df[f"bipower_variation_{short_w}d"]
    jump_intensity = safe_div(
        rv_mean - bpv_for_intensity, rv_mean, require_positive_denom=True
    ).clip(lower=0.0, upper=1.0)
    df[f"realized_jump_intensity_{short_w}d"] = jump_intensity

    feature_columns = list(feature_names)
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    output = df[["instrument_id", "date", *feature_columns]].copy()
    coverage = {name: int(output[name].notna().sum()) for name in feature_columns}
    return FeatureFrame(
        frame=output,
        feature_names=feature_names,
        feature_specs=spec_by_name,
        coverage=coverage,
        key_columns=DEFAULT_KEY_COLUMNS,
    )


__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_microstructure_features",
]
