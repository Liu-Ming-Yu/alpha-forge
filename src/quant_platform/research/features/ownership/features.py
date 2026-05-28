"""``ownership-v1`` feature factory.

Six features derived from institutional 13F holdings, FINRA
short-interest, and shares-outstanding records:

Institutional ownership (3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``institutional_ownership_pct`` —
  ``institutional_shares_total / shares_outstanding`` on date d.
  Range [0, 1] modulo the rare case where 13F totals exceed
  shares-out due to vendor data inconsistencies (clipped to 1.0).
* ``institutional_holder_count`` — distinct institutional filers
  reporting a position. Integer.
* ``institutional_ownership_change_qoq`` — per-instrument trailing
  change in ``institutional_ownership_pct`` over the configured
  quarter-over-quarter window (default 63 trading days). Positive
  = institutions accumulating; negative = institutions distributing.

Short interest (3)
~~~~~~~~~~~~~~~~~~

* ``short_interest_ratio`` —
  ``short_interest_shares / shares_outstanding``. Range [0, 1] modulo
  rare clipping. Higher = more shares sold short.
* ``days_to_cover`` —
  ``short_interest_shares / avg_daily_volume_shares``. Captures how
  many days the average daily volume would take to buy back the
  short. Common interpretation: > 5 = "crowded short".
* ``short_interest_change_4w`` — per-instrument trailing change in
  ``short_interest_ratio`` over the configured window (default 20
  trading days). Captures rising/falling short pressure.

Direction conventions and evidence gating
-----------------------------------------

All six features ship ``expected_direction="unknown"`` and
``larger_is_better=False`` — evidence-gated by construction.
Common a-priori intuitions (e.g. high short interest = bearish
contrarian signal, or institutional accumulation = bullish) are
empirically inconsistent across regimes. Promotion to a directional
spec is a family-version bump, not an in-place edit.

Data-feed status
----------------

v1 ships the **family scaffold** against explicit input dataclass
contracts. Real 13F and short-interest data feeds are not yet wired
into the platform (both are paid vendor products). Operator scripts
that populate the family from a vendor land in a follow-up; v1's
tests use synthetic fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.ownership.aggregator import build_ownership_panel
from quant_platform.research.features.ownership.config import (
    DEFAULT_CONFIG,
    OwnershipConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    group_by_instrument,
    group_shift,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.ownership.schemas import (
        Holding13FRecord,
        SharesOutstandingRecord,
        ShortInterestRecord,
    )


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
)


def _build_specs(
    version: str,
    *,
    holding_change_window_days: int,
    short_interest_change_window_days: int,
) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name="institutional_ownership_pct",
            family="ownership",
            description=(
                "Total institutional shares held (from 13F filings) divided "
                "by shares outstanding, on date d. Range [0, 1] modulo rare "
                "clipping when 13F totals exceed shares-out due to vendor "
                "data inconsistencies. PIT-safe: 13F rows enter the panel "
                "only on or after their 45-day-default availability lag."
            ),
            expected_direction="unknown",
            required_inputs=("institutional_shares_total", "shares_outstanding"),
            point_in_time=True,
            lookback_days=45,  # 13F availability lag
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="institutional_holder_count",
            family="ownership",
            description=(
                "Count of distinct institutional filers reporting a "
                "position on date d. Higher = broader institutional "
                "attention. Integer-valued."
            ),
            expected_direction="unknown",
            required_inputs=("institutional_holder_count",),
            point_in_time=True,
            lookback_days=45,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"institutional_ownership_change_{holding_change_window_days}d",
            family="ownership",
            description=(
                f"Per-instrument trailing change in institutional_ownership_pct "
                f"over the most recent {holding_change_window_days} trading "
                "days. Positive = institutions accumulating; negative = "
                "institutions distributing. Captures inter-filing drift in "
                "13F-derived ownership."
            ),
            expected_direction="unknown",
            required_inputs=("institutional_shares_total", "shares_outstanding"),
            point_in_time=True,
            lookback_days=45 + holding_change_window_days,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="short_interest_ratio",
            family="ownership",
            description=(
                "Short interest as a fraction of shares outstanding, on "
                "date d. Range [0, 1] modulo rare clipping. PIT-safe: "
                "short-interest records enter the panel only on or after "
                "their 8-day-default FINRA publication lag."
            ),
            expected_direction="unknown",
            required_inputs=("short_interest_shares", "shares_outstanding"),
            point_in_time=True,
            lookback_days=8,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="days_to_cover",
            family="ownership",
            description=(
                "Short interest divided by trailing average daily volume "
                "(shares). The number of trading days the average daily "
                "volume would take to buy back the entire short position. "
                "Common reading: > 5 = 'crowded short'."
            ),
            expected_direction="unknown",
            required_inputs=("short_interest_shares", "avg_daily_volume_shares"),
            point_in_time=True,
            lookback_days=8,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"short_interest_change_{short_interest_change_window_days}d",
            family="ownership",
            description=(
                f"Per-instrument trailing change in short_interest_ratio over "
                f"the most recent {short_interest_change_window_days} trading "
                "days. Positive = short pressure rising; negative = shorts "
                "covering."
            ),
            expected_direction="unknown",
            required_inputs=("short_interest_shares", "shares_outstanding"),
            point_in_time=True,
            lookback_days=8 + short_interest_change_window_days,
            version=version,
            larger_is_better=False,
        ),
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    holding_change_window_days=DEFAULT_CONFIG.holding_13f_change_window_days,
    short_interest_change_window_days=DEFAULT_CONFIG.short_interest_change_window_days,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


def _specs_for_config(config: OwnershipConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.holding_13f_change_window_days == DEFAULT_CONFIG.holding_13f_change_window_days
        and config.short_interest_change_window_days
        == DEFAULT_CONFIG.short_interest_change_window_days
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        holding_change_window_days=config.holding_13f_change_window_days,
        short_interest_change_window_days=config.short_interest_change_window_days,
    )


def compute_ownership_features(
    *,
    holdings: Iterable[Holding13FRecord],
    short_interest: Iterable[ShortInterestRecord],
    shares_outstanding: Iterable[SharesOutstandingRecord],
    trading_dates: pd.DatetimeIndex,
    config: OwnershipConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``ownership-v1`` feature panel.

    Parameters
    ----------
    holdings, short_interest, shares_outstanding:
        Input record iterables; see schemas for the contracts. Empty
        iterables are accepted — the family produces an all-NaN frame
        in that case rather than failing.
    trading_dates:
        The dates the panel must materialise rows for. Required —
        ownership is a per-day forward-filled signal, so the panel's
        row index is dictated by the calendar, not by the input
        records.
    config:
        :class:`OwnershipConfig`.
    """
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    holding_change_window = config.holding_13f_change_window_days
    si_change_window = config.short_interest_change_window_days

    panel = build_ownership_panel(
        holdings=holdings,
        short_interest=short_interest,
        shares_outstanding=shares_outstanding,
        trading_dates=trading_dates,
        holding_13f_availability_lag_days=config.holding_13f_availability_lag_days,
        short_interest_availability_lag_days=config.short_interest_availability_lag_days,
    ).frame

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

    shares_outstanding_series = panel["shares_outstanding"].astype(float)

    # institutional_ownership_pct = institutional_shares_total / shares_outstanding,
    # clipped at 1.0 (vendor data occasionally has 13F totals > shares-out
    # because of timing mismatches in fundamentals vs filings).
    inst_shares = panel["institutional_shares_total"].astype(float)
    panel["institutional_ownership_pct"] = safe_div(
        inst_shares, shares_outstanding_series, require_positive_denom=True
    ).clip(lower=0.0, upper=1.0)
    panel["institutional_holder_count"] = panel["institutional_holder_count"].astype(float)

    # institutional_ownership_change_<N>d: per-instrument trailing diff.
    grouped = group_by_instrument(panel)
    lagged_pct = group_shift(grouped["institutional_ownership_pct"], holding_change_window)
    panel[f"institutional_ownership_change_{holding_change_window}d"] = (
        panel["institutional_ownership_pct"] - lagged_pct
    )

    # short_interest_ratio = short_interest_shares / shares_outstanding.
    short_shares = panel["short_interest_shares"].astype(float)
    panel["short_interest_ratio"] = safe_div(
        short_shares, shares_outstanding_series, require_positive_denom=True
    ).clip(lower=0.0, upper=1.0)

    # days_to_cover = short_interest_shares / avg_daily_volume_shares.
    avg_volume = panel["avg_daily_volume_shares"].astype(float)
    panel["days_to_cover"] = safe_div(short_shares, avg_volume, require_positive_denom=True)

    # short_interest_change_<N>d: per-instrument trailing diff.
    grouped_si = group_by_instrument(panel)
    lagged_si = group_shift(grouped_si["short_interest_ratio"], si_change_window)
    panel[f"short_interest_change_{si_change_window}d"] = panel["short_interest_ratio"] - lagged_si

    output_columns = ["instrument_id", "date", *feature_names]
    output = panel[output_columns].copy()
    # Replace ±inf with NaN at the boundary (safe_div should never
    # produce them, but defence in depth).
    output[list(feature_names)] = output[list(feature_names)].replace([np.inf, -np.inf], np.nan)
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
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_ownership_features",
]
