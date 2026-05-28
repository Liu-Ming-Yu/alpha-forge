"""``macro-v1`` feature factory.

Six features derived from 8 FRED macro time series:

Yield-curve slopes (2)
~~~~~~~~~~~~~~~~~~~~~~

* ``yield_curve_slope_10y_2y`` — DGS10 − DGS2. Classic recession
  signal: persistent inversion historically precedes recessions by
  6–18 months.
* ``yield_curve_slope_10y_3m`` — DGS10 − DGS3MO. Fed-preferred curve
  metric (NY Fed's recession probability model uses this slope).

Credit risk (1)
~~~~~~~~~~~~~~~

* ``credit_spread_baa_aaa`` — Moody's Baa minus Aaa corporate yield.
  Widens in stress regimes; tighter in risk-on environments.

Equity vol (1)
~~~~~~~~~~~~~~

* ``vix_level`` — CBOE VIX close. Direct read of the canonical equity
  vol regime indicator.

FX momentum (1)
~~~~~~~~~~~~~~~

* ``dollar_index_change_30d`` — 30-calendar-day percent change in the
  Broad U.S. Dollar Index (DTWEXBGS). Positive = USD strengthening
  (risk-off proxy historically; also direct pressure on USD-denominated
  cross-border earnings).

Real rate (1)
~~~~~~~~~~~~~

* ``real_yield_10y`` — DFII10 direct read. The 10-year TIPS real
  yield. Distinct from the nominal yield because it strips out
  inflation expectations.

Direction conventions and evidence gating
-----------------------------------------

All six features ship ``expected_direction="unknown"`` and
``larger_is_better=False``. Macro indicators have well-known
historical signals (curve inversion → recession; high VIX → equity
weakness) but their predictive power on the SHORT-HORIZON forward
returns this platform optimises for is empirically inconsistent.
Promotion to a directional spec is a family-version bump.

Per-instrument broadcasting
---------------------------

Macro values are scalar per date. The compute function takes an
explicit ``instruments`` list and broadcasts the same per-date macro
value across all instruments to produce a standard
(instrument_id, date)-keyed :class:`FeatureFrame`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.macro.aggregator import build_macro_panel
from quant_platform.research.features.macro.config import (
    DEFAULT_CONFIG,
    FRED_CORPORATE_AAA,
    FRED_CORPORATE_BAA,
    FRED_DOLLAR_INDEX,
    FRED_TIPS_10Y,
    FRED_TREASURY_2Y,
    FRED_TREASURY_3M,
    FRED_TREASURY_10Y,
    FRED_VIX,
    REQUIRED_SERIES_IDS,
    MacroConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_platform.research.features.macro.schemas import MacroSeriesValue


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
)


def _build_specs(
    version: str,
    *,
    dollar_index_window_days: int,
) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name="yield_curve_slope_10y_2y",
            family="macro",
            description=(
                "10-year minus 2-year Treasury constant-maturity yield "
                "(FRED DGS10 − DGS2). Negative = inverted curve, a "
                "historically reliable 6-18 month leading indicator of "
                "U.S. recessions. Same per-date scalar broadcast across "
                "all instruments."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_TREASURY_10Y, FRED_TREASURY_2Y),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="yield_curve_slope_10y_3m",
            family="macro",
            description=(
                "10-year minus 3-month Treasury constant-maturity yield "
                "(FRED DGS10 − DGS3MO). The Fed-preferred curve metric — "
                "the NY Fed's recession-probability model uses this slope."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_TREASURY_10Y, FRED_TREASURY_3M),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="credit_spread_baa_aaa",
            family="macro",
            description=(
                "Moody's Baa minus Aaa corporate-bond yield spread (FRED "
                "BAA − AAA). Widens in stress regimes; tightens in risk-on "
                "environments. The granddaddy of credit-risk proxies."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_CORPORATE_BAA, FRED_CORPORATE_AAA),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="vix_level",
            family="macro",
            description=(
                "CBOE Volatility Index daily close (FRED VIXCLS). Direct "
                "read of the canonical equity-vol regime indicator. Same "
                "scalar broadcast across all instruments."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_VIX,),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"dollar_index_change_{dollar_index_window_days}d",
            family="macro",
            description=(
                f"{dollar_index_window_days}-calendar-day percent change in "
                "the Broad U.S. Dollar Index (FRED DTWEXBGS). Positive = "
                "USD strengthening; classic risk-off proxy historically. "
                "Also a direct pressure on USD-denominated cross-border "
                "earnings."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_DOLLAR_INDEX,),
            point_in_time=True,
            lookback_days=dollar_index_window_days,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="real_yield_10y",
            family="macro",
            description=(
                "10-year TIPS constant-maturity real yield (FRED DFII10). "
                "Distinct from the nominal yield because TIPS strip out "
                "inflation expectations — directly observes the real cost "
                "of capital. Direct read; no transformation."
            ),
            expected_direction="unknown",
            required_inputs=(FRED_TIPS_10Y,),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    dollar_index_window_days=DEFAULT_CONFIG.dollar_index_window_days,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


def _specs_for_config(config: MacroConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.dollar_index_window_days == DEFAULT_CONFIG.dollar_index_window_days
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        dollar_index_window_days=config.dollar_index_window_days,
    )


def compute_macro_features(
    *,
    series_values: Iterable[MacroSeriesValue],
    instruments: Sequence[str],
    trading_dates: pd.DatetimeIndex,
    config: MacroConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``macro-v1`` feature panel.

    Parameters
    ----------
    series_values:
        Iterable of :class:`MacroSeriesValue`. Series IDs not in
        :data:`REQUIRED_SERIES_IDS` are silently ignored; missing
        required series produce all-NaN columns in the output.
    instruments:
        List of instrument IDs to broadcast macro values across.
        The output frame has one row per (instrument, date).
    trading_dates:
        Calendar of dates.
    config:
        :class:`MacroConfig`.
    """
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    dollar_window = config.dollar_index_window_days

    macro_panel = build_macro_panel(
        series_values=series_values,
        trading_dates=trading_dates,
        required_series_ids=REQUIRED_SERIES_IDS,
    ).frame

    instruments_list = list(instruments)

    if not instruments_list or macro_panel.empty:
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

    # ------------------------------------------------------------------
    # Compute the per-date features from the macro panel.
    # ------------------------------------------------------------------
    macro = macro_panel.copy()
    macro["yield_curve_slope_10y_2y"] = macro[FRED_TREASURY_10Y].astype(float) - macro[
        FRED_TREASURY_2Y
    ].astype(float)
    macro["yield_curve_slope_10y_3m"] = macro[FRED_TREASURY_10Y].astype(float) - macro[
        FRED_TREASURY_3M
    ].astype(float)
    macro["credit_spread_baa_aaa"] = macro[FRED_CORPORATE_BAA].astype(float) - macro[
        FRED_CORPORATE_AAA
    ].astype(float)
    macro["vix_level"] = macro[FRED_VIX].astype(float)
    # Dollar-index momentum: percent change vs the value `window` calendar
    # days ago. Computed via merge_asof on a date-shifted copy of the
    # series — same trick estimates-v1 uses for lagged consensus.
    dollar_series = macro[["date", FRED_DOLLAR_INDEX]].dropna().copy()
    dollar_series["date_plus_window"] = dollar_series["date"] + pd.Timedelta(days=dollar_window)
    dollar_series = dollar_series.rename(columns={FRED_DOLLAR_INDEX: f"{FRED_DOLLAR_INDEX}_lag"})
    macro = pd.merge_asof(
        macro.sort_values("date", kind="stable").reset_index(drop=True),
        dollar_series[["date_plus_window", f"{FRED_DOLLAR_INDEX}_lag"]].sort_values(
            "date_plus_window", kind="stable"
        ),
        left_on="date",
        right_on="date_plus_window",
        direction="backward",
    ).drop(columns=["date_plus_window"])
    macro[f"dollar_index_change_{dollar_window}d"] = safe_div(
        macro[FRED_DOLLAR_INDEX].astype(float) - macro[f"{FRED_DOLLAR_INDEX}_lag"].astype(float),
        macro[f"{FRED_DOLLAR_INDEX}_lag"].astype(float),
        require_positive_denom=True,
    )
    macro["real_yield_10y"] = macro[FRED_TIPS_10Y].astype(float)

    # ------------------------------------------------------------------
    # Broadcast per-date features across the instrument list.
    # ------------------------------------------------------------------
    per_date_cols = ["date", *feature_names]
    per_date = macro[per_date_cols].copy()
    # Cross-join: one row per (instrument, date). pandas' `merge` with
    # a `how="cross"`-like construction via key=1 is the standard trick.
    instruments_frame = pd.DataFrame({"instrument_id": instruments_list, "_key": 1})
    per_date["_key"] = 1
    output = (
        instruments_frame.merge(per_date, on="_key", how="outer")
        .drop(columns=["_key"])
        .sort_values(["instrument_id", "date"])
        .reset_index(drop=True)
    )

    output[list(feature_names)] = output[list(feature_names)].replace([np.inf, -np.inf], np.nan)
    coverage = {name: int(output[name].notna().sum()) for name in feature_names}
    return FeatureFrame(
        frame=output[["instrument_id", "date", *feature_names]],
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
    "compute_macro_features",
]
