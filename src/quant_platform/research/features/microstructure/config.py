"""Configuration for the ``microstructure-v3`` feature set.

Ships **19 textbook microstructure proxies** computed from daily OHLCV
bars alone — designed to be **complementary to**, not overlap with,
``price-volume-starter-v1``. Tick-level features (Kyle's λ, VPIN,
true effective spread, order-flow imbalance) defer to a future
``microstructure-v4`` once a minute-bar or trade-tick feed is wired.

History:

* ``v1`` shipped 10 features (range-based vols, OHLC spread proxies,
  intraday position, serial dependence, volume coupling, range
  asymmetry).
* ``v2`` added 6 more: Yang-Zhang volatility, bipower variation,
  realized skew/kurtosis, Lo-MacKinlay variance ratio, range
  persistence. All still pure daily OHLCV.
* ``v3`` (this version) adds 3 **jump-cluster-robust** estimators
  motivated by a v2 test finding that bipower variation only resists
  *isolated* jumps — a jump cluster (e.g. price spike + mean
  reversion) defeats it. The three additions close the limitation:

  - ``med_rv_<long>d``                    Median Realized Variance
                                          (Andersen-Dobrev-Schaumburg
                                          2012) — uses the median of
                                          3 adjacent ``|r|`` values,
                                          so any single-day jump is
                                          fully neutralised.
  - ``tripower_variation_<short>d``       Tripower variation
                                          (Barndorff-Nielsen-Shephard
                                          2006) — three-return
                                          generalisation of bipower
                                          with exponent 2/3, more
                                          robust to multi-day jump
                                          clusters than BPV.
  - ``realized_jump_intensity_<short>d``  Andersen-Bollerslev-Diebold
                                          (2007) jump intensity:
                                          ``(naive_RV - BPV) / RV``
                                          clipped to [0, 1]. Turns
                                          the v2 BPV-vs-RV gap into
                                          a positive signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Family version. Bump on formula change, input addition, or
#: lookback-window rename. ``v3`` adds 3 jump-cluster-robust
#: estimators (MedRV, tripower variation, realized jump intensity)
#: on top of v2's 16 features. Tick/quote features will land in a
#: future ``microstructure-v4`` once intraday data is wired.
FEATURE_SET_VERSION: str = "microstructure-v3"

#: Default short rolling window (≈ 1 month of trading days). Used by
#: range-based volatility estimators, close-in-range, volume/return
#: correlation, Corwin-Schultz spread, Yang-Zhang volatility,
#: bipower variation, range persistence.
DEFAULT_SHORT_WINDOW: int = 20

#: Default long rolling window (≈ 3 months). Used by Roll's spread
#: estimator, return/volume autocorrelation, realized higher moments
#: (skew/kurt), and the variance ratio.
DEFAULT_LONG_WINDOW: int = 60

#: Multi-day return stride for the Lo-MacKinlay variance ratio.
#: VR(q) = Var(q-day returns) / (q × Var(1-day returns)). Convention is
#: ``q = 5`` (weekly), which matches the original 1988 paper's lead
#: example and gives a meaningful sample even at 60-day rolling
#: windows.
DEFAULT_VARIANCE_RATIO_STRIDE: int = 5


@dataclass(frozen=True)
class MicrostructureConfig(BaseFamilyConfig):
    """Frozen config for the microstructure feature factory.

    ``lookback_days`` convention
    ----------------------------
    Every :class:`FeatureSpec` in this family sets ``lookback_days``
    to **the first row at which the feature is guaranteed non-NaN**,
    *including* warm-up costs from shift / lag operations:

    * Rolling-only features → ``lookback_days = window``.
    * Features that need a lag (overnight, bipower, range persistence)
      → ``lookback_days = window + 1``.
    * Variance ratio (rolling var of q-stride returns) →
      ``lookback_days = window + q``.

    This is the "first non-NaN row" interpretation rather than the
    bare "rolling window" interpretation; consumers using
    ``lookback_days`` to pre-warm walk-forward history don't have to
    add a separate fudge factor.

    Attributes
    ----------
    version:
        Family-version string stamped into every emitted
        :class:`FeatureSpec`. Defaults to :data:`FEATURE_SET_VERSION`.
        Bumping any of the window/stride knobs below WITHOUT also
        bumping ``version`` is rejected at compute time — see
        :func:`._enforce_version_window_consistency`.
    short_window:
        Trading-day window for the short-horizon features
        (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang,
        close-in-range, Corwin-Schultz, volume-return correlation,
        high-low asymmetry, bipower variation, range persistence).
        Bumping this requires a feature-set version bump because the
        window appears in feature column names.

        Lower-bounded at 5: Yang-Zhang's k-weight denominator is
        ``(short_window - 1)``, which would blow up at ``short_window
        = 1``; the 5-day floor leaves comfortable headroom. The
        ``compute`` path also asserts ``short_window >= 2`` as a
        belt-and-braces guard.
    long_window:
        Trading-day window for the slower signals (Roll's spread,
        return autocorrelation, volume autocorrelation, realized
        skew/kurt, variance ratio). Same versioning rule.
    variance_ratio_stride:
        ``q`` in the Lo-MacKinlay VR(q) statistic. Default 5 days.
        Bumping changes the feature column name
        (``variance_ratio_q_1_<long>d``) so it requires a version bump.
    """

    version: str = FEATURE_SET_VERSION
    short_window: int = DEFAULT_SHORT_WINDOW
    long_window: int = DEFAULT_LONG_WINDOW
    variance_ratio_stride: int = DEFAULT_VARIANCE_RATIO_STRIDE

    def __post_init__(self) -> None:
        super().__post_init__()
        # ``short_window >= 5`` also protects the Yang-Zhang k-weight
        # denominator ``(short_window - 1)`` from divide-by-zero / very
        # small denominators on the noise floor.
        if self.short_window < 5:
            raise ValueError("MicrostructureConfig.short_window must be >= 5")
        if self.long_window < 10:
            raise ValueError("MicrostructureConfig.long_window must be >= 10")
        if self.long_window <= self.short_window:
            raise ValueError(
                "MicrostructureConfig.long_window must be strictly greater than short_window"
            )
        if self.variance_ratio_stride < 2:
            raise ValueError("MicrostructureConfig.variance_ratio_stride must be >= 2")
        if self.variance_ratio_stride >= self.long_window:
            raise ValueError(
                "MicrostructureConfig.variance_ratio_stride must be < long_window so the "
                "long window can hold enough q-stride returns to estimate var"
            )


DEFAULT_CONFIG: MicrostructureConfig = MicrostructureConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_LONG_WINDOW",
    "DEFAULT_SHORT_WINDOW",
    "DEFAULT_VARIANCE_RATIO_STRIDE",
    "FEATURE_SET_VERSION",
    "MicrostructureConfig",
]
