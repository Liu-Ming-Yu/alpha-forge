"""``price-volume-starter-v1`` feature factory.

Computes OHLCV-derived features at the ``(instrument_id, date)`` grain. All
features are computed after the close of ``date`` and must only be used for
labels or decisions strictly after that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureFrame,
    FeatureSpec,
)
from quant_platform.services.research_service.features.kernel.price_volume.config import (
    DEFAULT_CONFIG,
    LOOKBACK_52W_HIGH,
    LOOKBACK_AMIHUD,
    LOOKBACK_DOLLAR_VOLUME,
    LOOKBACK_HIGH_LOW_RANGE,
    LOOKBACK_VOLUME_ZSCORE,
    LOOKBACKS_RETURN,
    LOOKBACKS_VOL,
    PriceVolumeConfig,
)
from quant_platform.services.research_service.features.kernel.transforms import (
    DEFAULT_KEY_COLUMNS,
    TRADING_DAYS_PER_MONTH,
    group_by_instrument,
    group_rolling_max,
    group_rolling_mean,
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

#: Deprecated alias kept so old callers do not break. New code should
#: read ``REQUIRED_INPUT_COLUMNS``; this alias will be removed at the
#: next feature-set version bump.
REQUIRED_BAR_COLUMNS: tuple[str, ...] = REQUIRED_INPUT_COLUMNS


def _ret_spec(window: int, version: str) -> FeatureSpec:
    return FeatureSpec(
        name=f"ret_{window}d",
        family="price_volume",
        description=(
            f"{window}-trading-day total return: close / close.shift({window}) - 1. "
            "Positive-oriented under the momentum prior at horizons >= 21d; "
            "the short-horizon ret_1d/5d are evidence-uncertain and are also "
            "surfaced under explicit reversal_* names."
        ),
        expected_direction="unknown" if window <= 5 else "+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=window,
        version=version,
        signal_timestamp="eod_after_close",
        larger_is_better=window > 5,
    )


def _mom_spec(month: int, version: str) -> FeatureSpec:
    long_lb = {3: 63, 6: 126, 12: 252}[month]
    return FeatureSpec(
        name=f"mom_{month}_1",
        family="price_volume",
        description=(
            f"{month}-month momentum excluding the most recent month: "
            f"close.shift(21) / close.shift({long_lb}) - 1. Excluding the "
            "most recent 21d avoids short-term reversal contamination."
        ),
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=long_lb,
        version=version,
        signal_timestamp="eod_after_close",
    )


def _reversal_spec(window: int, version: str) -> FeatureSpec:
    return FeatureSpec(
        name=f"reversal_{window}d",
        family="price_volume",
        description=(
            f"Sign-flipped {window}d return: -(close / close.shift({window}) - 1). "
            "Larger value = stronger recent underperformance, expecting reversal."
        ),
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=window,
        version=version,
        signal_timestamp="eod_after_close",
    )


def _low_vol_spec(window: int, version: str, *, downside: bool = False) -> FeatureSpec:
    name = f"low_downside_vol_{window}d" if downside else f"low_vol_{window}d"
    formula = (
        f"-rolling_std(min(daily_return, 0), {window})"
        if downside
        else f"-rolling_std(daily_return, {window})"
    )
    return FeatureSpec(
        name=name,
        family="price_volume",
        description=(
            f"Sign-flipped {window}d "
            f"{'downside ' if downside else ''}realized volatility: {formula}. "
            "Larger value = lower realized risk."
        ),
        expected_direction="+",
        required_inputs=("close",),
        point_in_time=True,
        lookback_days=window,
        version=version,
        signal_timestamp="eod_after_close",
    )


def _build_specs(version: str) -> tuple[FeatureSpec, ...]:
    """Construct every spec for a price-volume feature-set version."""
    specs: list[FeatureSpec] = []

    for window in LOOKBACKS_RETURN:
        specs.append(_ret_spec(window, version))

    for month in (12, 6, 3):
        specs.append(_mom_spec(month, version))

    for window in (1, 5, 21):
        specs.append(_reversal_spec(window, version))

    for window in LOOKBACKS_VOL:
        specs.append(_low_vol_spec(window, version))
    specs.append(_low_vol_spec(63, version, downside=True))

    specs.append(
        FeatureSpec(
            name="volume_z_20d",
            family="price_volume",
            description=(
                "20-day z-score of daily share volume: "
                "(volume - rolling_mean(volume, 20)) / rolling_std(volume, 20). "
                "Sign in the cross-section is regime-dependent."
            ),
            expected_direction="unknown",
            required_inputs=("volume",),
            point_in_time=True,
            lookback_days=LOOKBACK_VOLUME_ZSCORE,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
    )
    specs.append(
        FeatureSpec(
            name="dollar_volume_20d",
            family="price_volume",
            description=(
                "20-day mean dollar volume: rolling_mean(close * volume, 20). "
                "Liquidity proxy used both as a feature and as a denominator in "
                "low_amihud_20d."
            ),
            expected_direction="+",
            required_inputs=("close", "volume"),
            point_in_time=True,
            lookback_days=LOOKBACK_DOLLAR_VOLUME,
            version=version,
            signal_timestamp="eod_after_close",
        )
    )
    specs.append(
        FeatureSpec(
            name="low_amihud_20d",
            family="price_volume",
            description=(
                "Sign-flipped 20-day Amihud illiquidity: "
                "-rolling_mean(|daily_return| / dollar_volume, 20). "
                "Larger value = more liquid."
            ),
            expected_direction="+",
            required_inputs=("close", "volume"),
            point_in_time=True,
            lookback_days=LOOKBACK_AMIHUD,
            version=version,
            signal_timestamp="eod_after_close",
        )
    )
    specs.append(
        FeatureSpec(
            name="high_low_range_1d",
            family="price_volume",
            description="(high - low) / close. Same-day realized range proxy.",
            expected_direction="unknown",
            required_inputs=("high", "low", "close"),
            point_in_time=True,
            lookback_days=0,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
    )
    specs.append(
        FeatureSpec(
            name="high_low_range_20d",
            family="price_volume",
            description="rolling_mean(high_low_range_1d, 20).",
            expected_direction="unknown",
            required_inputs=("high", "low", "close"),
            point_in_time=True,
            lookback_days=LOOKBACK_HIGH_LOW_RANGE,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
    )
    specs.append(
        FeatureSpec(
            name="overnight_gap",
            family="price_volume",
            description="open / close.shift(1) - 1.",
            expected_direction="unknown",
            required_inputs=("open", "close"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
    )
    specs.append(
        FeatureSpec(
            name="open_to_close_return",
            family="price_volume",
            description="close / open - 1. Intra-session return.",
            expected_direction="unknown",
            required_inputs=("open", "close"),
            point_in_time=True,
            lookback_days=0,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
    )
    specs.append(
        FeatureSpec(
            name="close_to_open_return",
            family="price_volume",
            description=(
                "open / close.shift(1) - 1. Identical formula to overnight_gap; "
                "kept as catalog alias until evidence picks a winner."
            ),
            expected_direction="unknown",
            required_inputs=("open", "close"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
            canonical_name="overnight_gap",
        )
    )
    specs.append(
        FeatureSpec(
            name="distance_to_52w_high",
            family="price_volume",
            description=(
                "close / rolling_max(close, 252) - 1. Negative or zero; "
                "closer to zero = closer to 52-week high."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=LOOKBACK_52W_HIGH,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
            aliases=("drawdown_from_252d_high",),
        )
    )
    specs.append(
        FeatureSpec(
            name="drawdown_from_252d_high",
            family="price_volume",
            description=(
                "close / rolling_max(close, 252) - 1. Same formula as "
                "distance_to_52w_high; preserved as a catalog alias."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=LOOKBACK_52W_HIGH,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
            canonical_name="distance_to_52w_high",
        )
    )

    return tuple(specs)


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(DEFAULT_CONFIG.version)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)
_SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}


def _validate_inputs(bars: pd.DataFrame) -> None:
    """Reject bar frames that are missing required OHLCV columns."""
    missing = [name for name in REQUIRED_INPUT_COLUMNS if name not in bars.columns]
    if missing:
        raise ValueError(
            "compute_price_volume_features: bars missing required columns: "
            f"{missing!r}; got {list(bars.columns)!r}"
        )


def _specs_for_config(config: PriceVolumeConfig) -> tuple[FeatureSpec, ...]:
    if config.version == DEFAULT_CONFIG.version:
        return FEATURE_SPECS
    return _build_specs(config.version)


def compute_price_volume_features(
    bars: pd.DataFrame,
    *,
    config: PriceVolumeConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the price-volume feature panel."""
    _validate_inputs(bars)
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}

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

    grouped = group_by_instrument(df)
    grp_close = grouped["close"]
    grp_volume = grouped["volume"]
    policy = config.min_periods_policy

    close_lag1 = group_shift(grp_close, 1)
    daily_ret = safe_div(df["close"], close_lag1) - 1.0
    df["_daily_ret"] = daily_ret
    grp_ret = group_by_instrument(df)["_daily_ret"]

    for window in LOOKBACKS_RETURN:
        df[f"ret_{window}d"] = safe_div(df["close"], group_shift(grp_close, window)) - 1.0

    close_lag21 = group_shift(grp_close, TRADING_DAYS_PER_MONTH)
    for month, long_lb in ((12, 252), (6, 126), (3, 63)):
        df[f"mom_{month}_1"] = safe_div(close_lag21, group_shift(grp_close, long_lb)) - 1.0

    for window in (1, 5, 21):
        df[f"reversal_{window}d"] = -df[f"ret_{window}d"]

    for window in LOOKBACKS_VOL:
        df[f"low_vol_{window}d"] = -group_rolling_std(
            grp_ret,
            window,
            policy=policy,
        )

    df["_downside_ret"] = df["_daily_ret"].clip(upper=0.0)
    grp_downside = group_by_instrument(df)["_downside_ret"]
    df["low_downside_vol_63d"] = -group_rolling_std(
        grp_downside,
        63,
        policy=policy,
    )

    df["_dollar_volume"] = df["close"] * df["volume"]
    grp_dv = group_by_instrument(df)["_dollar_volume"]
    df["dollar_volume_20d"] = group_rolling_mean(
        grp_dv,
        LOOKBACK_DOLLAR_VOLUME,
        policy=policy,
    )

    vol_mean_20 = group_rolling_mean(
        grp_volume,
        LOOKBACK_VOLUME_ZSCORE,
        policy=policy,
    )
    vol_std_20 = group_rolling_std(
        grp_volume,
        LOOKBACK_VOLUME_ZSCORE,
        policy=policy,
    )
    df["volume_z_20d"] = safe_div(df["volume"] - vol_mean_20, vol_std_20)

    df["_amihud_daily"] = safe_div(df["_daily_ret"].abs(), df["_dollar_volume"])
    grp_amihud = group_by_instrument(df)["_amihud_daily"]
    df["low_amihud_20d"] = -group_rolling_mean(
        grp_amihud,
        LOOKBACK_AMIHUD,
        policy=policy,
    )

    df["high_low_range_1d"] = safe_div(df["high"] - df["low"], df["close"])
    grp_hl = group_by_instrument(df)["high_low_range_1d"]
    df["high_low_range_20d"] = group_rolling_mean(
        grp_hl,
        LOOKBACK_HIGH_LOW_RANGE,
        policy=policy,
    )
    df["overnight_gap"] = safe_div(df["open"], close_lag1) - 1.0
    df["open_to_close_return"] = safe_div(df["close"], df["open"]) - 1.0
    df["close_to_open_return"] = safe_div(df["open"], close_lag1) - 1.0

    rolling_high_252 = group_rolling_max(
        grp_close,
        LOOKBACK_52W_HIGH,
        policy=policy,
    )
    distance = safe_div(df["close"], rolling_high_252) - 1.0
    df["distance_to_52w_high"] = distance
    df["drawdown_from_252d_high"] = distance

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
    "REQUIRED_BAR_COLUMNS",  # deprecated alias
    "REQUIRED_INPUT_COLUMNS",
    "compute_price_volume_features",
]
