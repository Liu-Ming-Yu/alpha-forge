"""``options-v1`` feature factory.

Six features derived from daily options-implied snapshots:

Volatility surface (3)
~~~~~~~~~~~~~~~~~~~~~~

* ``iv_30d_atm`` — at-the-money implied vol at the 30-day expiry.
  CBOE-standard reference (same expiry the VIX is calculated against).
* ``iv_skew_25d`` — ``iv_25d_put − iv_25d_call``. Positive = puts
  trading rich relative to calls (downside-hedging premium).
* ``iv_term_slope`` — ``(iv_60d_atm − iv_30d_atm) / iv_30d_atm``.
  Positive = contango (longer-dated vol higher, calm markets);
  negative = backwardation (short-dated vol higher, stress regime).

Activity (2)
~~~~~~~~~~~~

* ``put_call_volume_ratio`` — ``put_volume / call_volume``. > 1 =
  more put volume; classic contrarian sentiment proxy.
* ``put_call_oi_ratio`` — same with open interest. Slower-moving
  positioning signal vs the volume version.

IV vs realized (1)
~~~~~~~~~~~~~~~~~~

* ``iv_realized_premium_30d`` — ``iv_30d_atm − realized_vol_21d``.
  Positive = options pricing in more vol than has been realized
  (vol risk premium captured).

Direction conventions and evidence gating
-----------------------------------------

All six features ship ``expected_direction="unknown"`` and
``larger_is_better=False`` — evidence-gated. The options-implied
literature is rich in conflicting findings (skew as crash insurance
vs. behavioral overpricing; VRP as risk premium vs. mean-reverting
predictor). Promotion to a directional spec is a family-version bump.

Data-feed status
----------------

v1 ships the family scaffold against the :class:`OptionsSnapshot`
input contract. Real options-chain feed wiring (CBOE, OptionMetrics,
Polygon options, ORATS) is a separate (out-of-scope-for-v1) PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.options.aggregator import build_options_panel
from quant_platform.research.features.options.config import (
    DEFAULT_CONFIG,
    OptionsConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.options.schemas import OptionsSnapshot


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
)


def _build_specs(
    version: str,
    *,
    atm_tenor_days: int,
    term_long_tenor_days: int,
    realized_vol_window_days: int,
) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name=f"iv_{atm_tenor_days}d_atm",
            family="options",
            description=(
                f"At-the-money implied volatility at the {atm_tenor_days}-day "
                "expiry. Derived from the vendor's interpolated options "
                "surface. The CBOE-standard reference point for IV is the "
                "30-day expiry — same tenor the VIX is calculated against."
            ),
            expected_direction="unknown",
            required_inputs=(f"iv_{atm_tenor_days}d_atm",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="iv_skew_25d",
            family="options",
            description=(
                "25-delta implied-vol skew: iv_25d_put - iv_25d_call. "
                "Positive = OTM puts trading richer than OTM calls "
                "(classic downside-hedging premium); negative = OTM calls "
                "trading richer (upside-chase premium). NaN when either "
                "25Δ leg is missing from the vendor surface."
            ),
            expected_direction="unknown",
            required_inputs=("iv_25d_call", "iv_25d_put"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="iv_term_slope",
            family="options",
            description=(
                f"IV term structure slope: (iv_{term_long_tenor_days}d_atm - "
                f"iv_{atm_tenor_days}d_atm) / iv_{atm_tenor_days}d_atm. "
                "Positive = contango (longer-dated vol higher than near-dated; "
                "calm regime); negative = backwardation (short-dated vol "
                "premium; stress regime). NaN when either tenor is missing or "
                f"iv_{atm_tenor_days}d_atm is zero."
            ),
            expected_direction="unknown",
            required_inputs=(
                f"iv_{atm_tenor_days}d_atm",
                f"iv_{term_long_tenor_days}d_atm",
            ),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="put_call_volume_ratio",
            family="options",
            description=(
                "Daily put_volume / call_volume across all strikes and "
                "expiries. Values > 1 = more put volume traded than call. "
                "Classic contrarian-sentiment proxy: extreme highs are "
                "associated with bearish positioning that historically marks "
                "near-term troughs (mean reversion) — but evidence is "
                "regime-dependent. NaN when call_volume is zero."
            ),
            expected_direction="unknown",
            required_inputs=("put_volume", "call_volume"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="put_call_oi_ratio",
            family="options",
            description=(
                "put_open_interest / call_open_interest. Same intuition as "
                "the volume ratio but slower-moving — captures position-level "
                "rather than flow-level skew. NaN when call_open_interest is "
                "zero."
            ),
            expected_direction="unknown",
            required_inputs=("put_open_interest", "call_open_interest"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"iv_realized_premium_{atm_tenor_days}d",
            family="options",
            description=(
                f"iv_{atm_tenor_days}d_atm minus the trailing "
                f"{realized_vol_window_days}-trading-day realized volatility "
                "of the underlying. Positive = options pricing more vol than "
                "has been realized (vol risk premium captured); negative = "
                "realized vol exceeds implied (under-priced; could indicate "
                "regime shift)."
            ),
            expected_direction="unknown",
            required_inputs=(f"iv_{atm_tenor_days}d_atm", "realized_vol_21d"),
            point_in_time=True,
            lookback_days=realized_vol_window_days,
            version=version,
            larger_is_better=False,
        ),
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    atm_tenor_days=DEFAULT_CONFIG.atm_tenor_days,
    term_long_tenor_days=DEFAULT_CONFIG.term_long_tenor_days,
    realized_vol_window_days=DEFAULT_CONFIG.realized_vol_window_days,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


def _specs_for_config(config: OptionsConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.atm_tenor_days == DEFAULT_CONFIG.atm_tenor_days
        and config.term_long_tenor_days == DEFAULT_CONFIG.term_long_tenor_days
        and config.realized_vol_window_days == DEFAULT_CONFIG.realized_vol_window_days
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        atm_tenor_days=config.atm_tenor_days,
        term_long_tenor_days=config.term_long_tenor_days,
        realized_vol_window_days=config.realized_vol_window_days,
    )


def compute_options_features(
    *,
    snapshots: Iterable[OptionsSnapshot],
    trading_dates: pd.DatetimeIndex,
    config: OptionsConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``options-v1`` feature panel.

    Parameters
    ----------
    snapshots:
        Iterable of :class:`OptionsSnapshot`.
    trading_dates:
        Calendar of dates the panel materialises rows for.
    config:
        :class:`OptionsConfig`.
    """
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    atm = config.atm_tenor_days
    long_tenor = config.term_long_tenor_days

    panel = build_options_panel(snapshots=snapshots, trading_dates=trading_dates).frame

    if panel.empty:
        empty = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "date": pd.Series(dtype="datetime64[ns]"),
                **{name: pd.Series(dtype=float) for name in feature_names},
            }
        )
        return FeatureFrame(
            frame=empty,
            feature_names=feature_names,
            feature_specs=spec_by_name,
            coverage={name: 0 for name in feature_names},
            key_columns=DEFAULT_KEY_COLUMNS,
        )

    panel = panel.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # iv_30d_atm: direct read.
    panel[f"iv_{atm}d_atm"] = panel["iv_30d_atm"].astype(float)

    # iv_skew_25d: 25Δ put IV − 25Δ call IV. NaN-propagating.
    panel["iv_skew_25d"] = panel["iv_25d_put"].astype(float) - panel["iv_25d_call"].astype(float)

    # iv_term_slope: (long - short) / short. safe_div for the zero
    # short-tenor IV edge case.
    iv_short = panel["iv_30d_atm"].astype(float)
    iv_long = panel["iv_60d_atm"].astype(float)
    panel["iv_term_slope"] = safe_div(iv_long - iv_short, iv_short, require_positive_denom=True)

    # Put-call ratios.
    panel["put_call_volume_ratio"] = safe_div(
        panel["put_volume"].astype(float),
        panel["call_volume"].astype(float),
        require_positive_denom=True,
    )
    panel["put_call_oi_ratio"] = safe_div(
        panel["put_open_interest"].astype(float),
        panel["call_open_interest"].astype(float),
        require_positive_denom=True,
    )

    # iv_realized_premium: iv_30d_atm - realized_vol_21d.
    panel[f"iv_realized_premium_{atm}d"] = panel["iv_30d_atm"].astype(float) - panel[
        "realized_vol_21d"
    ].astype(float)

    output = panel[["instrument_id", "date", *feature_names]].copy()
    output[list(feature_names)] = output[list(feature_names)].replace([np.inf, -np.inf], np.nan)
    coverage = {name: int(output[name].notna().sum()) for name in feature_names}
    # Silence unused-variable warnings for `long_tenor`; the value is
    # consumed by `_build_specs` via the config, not directly inline.
    _ = long_tenor
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
    "compute_options_features",
]
