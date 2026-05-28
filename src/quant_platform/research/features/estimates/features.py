"""``estimates-v1`` feature factory.

Six features derived from daily analyst-consensus snapshots + historical
earnings-surprise records:

Revision magnitude (2)
~~~~~~~~~~~~~~~~~~~~~~

* ``eps_estimate_revision_30d`` —
  ``(eps_mean[T] - eps_mean[T - 30 cal days]) / |eps_mean[T - 30]|``.
  Relative drift in the FY1 EPS consensus.
* ``revenue_estimate_revision_30d`` — same shape, for FY1 revenue.

Revision direction (1)
~~~~~~~~~~~~~~~~~~~~~~

* ``eps_estimate_up_vs_down_30d`` —
  ``(n_up - n_down) / (n_up + n_down)``, ∈ [-1, 1]. Captures the
  balance of recent analyst opinion shifts. NaN when no analyst
  revised either way in the window.

Uncertainty + coverage (2)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``eps_estimate_dispersion`` — ``std(estimates) / |mean(estimates)|``
  on date d. The coefficient of variation; higher = more analyst
  disagreement. NaN for single-analyst coverage (std undefined) or
  zero consensus mean.
* ``analyst_coverage_count`` — number of analysts contributing to the
  FY1 EPS consensus on date d. Integer; broader = more institutional
  attention.

Surprise history (1)
~~~~~~~~~~~~~~~~~~~~

* ``eps_surprise_mean_4q`` — mean % surprise over the most recent
  configured-N reported quarters (default 4). Captures whether the
  company has been beating or missing consensus systematically.

Direction conventions and evidence gating
-----------------------------------------

All six features ship ``expected_direction="unknown"`` and
``larger_is_better=False`` — evidence-gated. The estimate-revision
literature has decades of conflicting findings on signs (PEAD vs
overreaction; surprise drift vs reversal). Promotion to a directional
spec is a family-version bump.

Data-feed status
----------------

v1 ships the **family scaffold** against explicit input dataclass
contracts (:class:`ConsensusSnapshot`, :class:`EarningsSurpriseRecord`).
Real IBES / FactSet / Visible Alpha feed wiring is a separate
(out-of-scope-for-v1) PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.estimates.aggregator import build_estimates_panel
from quant_platform.research.features.estimates.config import (
    DEFAULT_CONFIG,
    EstimatesConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.estimates.schemas import (
        ConsensusSnapshot,
        EarningsSurpriseRecord,
    )


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
)


def _build_specs(
    version: str,
    *,
    revision_window_days: int,
    surprise_lookback_quarters: int,
) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name=f"eps_estimate_revision_{revision_window_days}d",
            family="estimates",
            description=(
                f"Relative change in the FY1 EPS consensus mean over the "
                f"trailing {revision_window_days} calendar days: "
                "(mean[T] - mean[T - window]) / |mean[T - window]|. "
                "Positive = analysts collectively raising their EPS "
                "estimates; negative = lowering. NaN when the lagged "
                "consensus is zero (no denominator)."
            ),
            expected_direction="unknown",
            required_inputs=("eps_mean", f"eps_mean_lag_{revision_window_days}"),
            point_in_time=True,
            lookback_days=revision_window_days,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="eps_estimate_up_vs_down_30d",
            family="estimates",
            description=(
                "Balance of analyst revisions in the trailing 30 days: "
                "(n_up - n_down) / (n_up + n_down). Range [-1, 1]. +1 = "
                "all revisions are upward; -1 = all downward; 0 = "
                "balanced. NaN when no analyst revised either way."
            ),
            expected_direction="unknown",
            required_inputs=("eps_n_up_30d", "eps_n_down_30d"),
            point_in_time=True,
            lookback_days=30,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="eps_estimate_dispersion",
            family="estimates",
            description=(
                "Coefficient of variation across analyst FY1 EPS "
                "estimates on date d: std(estimates) / |mean(estimates)|. "
                "Higher = more analyst disagreement (uncertainty proxy). "
                "NaN when only one analyst covers (std undefined) or the "
                "mean is exactly zero."
            ),
            expected_direction="unknown",
            required_inputs=("eps_mean", "eps_std"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="analyst_coverage_count",
            family="estimates",
            description=(
                "Number of analysts contributing to the FY1 EPS consensus "
                "on date d. Integer-valued, non-negative. Broader = more "
                "institutional attention."
            ),
            expected_direction="unknown",
            required_inputs=("eps_n",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"eps_surprise_mean_{surprise_lookback_quarters}q",
            family="estimates",
            description=(
                f"Mean per-quarter percent EPS surprise across the most "
                f"recent {surprise_lookback_quarters} reported fiscal "
                "periods: ((actual - consensus) / |consensus|) averaged. "
                "Positive = company has been beating consensus on average; "
                "negative = missing. NaN before any actuals report on or "
                "after the panel start date (PIT-safe: surprises only "
                "enter the panel after their reported_at)."
            ),
            expected_direction="unknown",
            required_inputs=("eps_surprise_mean_recent",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=f"revenue_estimate_revision_{revision_window_days}d",
            family="estimates",
            description=(
                f"Relative change in the FY1 revenue consensus mean over "
                f"the trailing {revision_window_days} calendar days. Same "
                "formula as eps_estimate_revision but for sales rather "
                "than earnings. NaN when the lagged consensus is zero."
            ),
            expected_direction="unknown",
            required_inputs=(
                "revenue_mean",
                f"revenue_mean_lag_{revision_window_days}",
            ),
            point_in_time=True,
            lookback_days=revision_window_days,
            version=version,
            larger_is_better=False,
        ),
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    revision_window_days=DEFAULT_CONFIG.revision_window_days,
    surprise_lookback_quarters=DEFAULT_CONFIG.surprise_lookback_quarters,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)


def _specs_for_config(config: EstimatesConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.revision_window_days == DEFAULT_CONFIG.revision_window_days
        and config.surprise_lookback_quarters == DEFAULT_CONFIG.surprise_lookback_quarters
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        revision_window_days=config.revision_window_days,
        surprise_lookback_quarters=config.surprise_lookback_quarters,
    )


def compute_estimate_features(
    *,
    consensus_snapshots: Iterable[ConsensusSnapshot],
    surprise_records: Iterable[EarningsSurpriseRecord],
    trading_dates: pd.DatetimeIndex,
    config: EstimatesConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``estimates-v1`` feature panel.

    Parameters
    ----------
    consensus_snapshots:
        Iterable of :class:`ConsensusSnapshot`. Records whose
        target_period / estimate_kind doesn't match the configured
        EPS / revenue targets are silently filtered out.
    surprise_records:
        Iterable of :class:`EarningsSurpriseRecord`. Masked by
        ``reported_at <= panel_date`` for PIT-safety.
    trading_dates:
        Calendar of dates the panel materialises rows for. Required.
    config:
        :class:`EstimatesConfig`.
    """
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    revision_window = config.revision_window_days
    surprise_lookback = config.surprise_lookback_quarters

    panel = build_estimates_panel(
        consensus_snapshots=consensus_snapshots,
        surprise_records=surprise_records,
        trading_dates=trading_dates,
        eps_target_period=config.eps_target_period,
        revenue_target_period=config.revenue_target_period,
        revision_window_days=revision_window,
        surprise_lookback_quarters=surprise_lookback,
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

    # eps_estimate_revision_<window>d: relative change in mean consensus
    # over the window. NaN when the lagged consensus is zero (no
    # meaningful denominator).
    eps_mean = panel["eps_mean"].astype(float)
    eps_mean_lag = panel[f"eps_mean_lag_{revision_window}"].astype(float)
    panel[f"eps_estimate_revision_{revision_window}d"] = safe_div(
        eps_mean - eps_mean_lag,
        eps_mean_lag.abs(),
        require_positive_denom=True,
    )

    # eps_estimate_up_vs_down_30d: (up - down) / (up + down).
    n_up = panel["eps_n_up_30d"].astype(float)
    n_down = panel["eps_n_down_30d"].astype(float)
    panel["eps_estimate_up_vs_down_30d"] = safe_div(
        n_up - n_down, n_up + n_down, require_positive_denom=True
    )

    # eps_estimate_dispersion: std / |mean|.
    eps_std = panel["eps_std"].astype(float)
    panel["eps_estimate_dispersion"] = safe_div(
        eps_std, eps_mean.abs(), require_positive_denom=True
    )

    # analyst_coverage_count: direct read.
    panel["analyst_coverage_count"] = panel["eps_n"].astype(float)

    # eps_surprise_mean_<N>q: direct read from the surprise stream.
    panel[f"eps_surprise_mean_{surprise_lookback}q"] = panel["eps_surprise_mean_recent"].astype(
        float
    )

    # revenue_estimate_revision_<window>d: same shape as EPS.
    rev_mean = panel["revenue_mean"].astype(float)
    rev_mean_lag = panel[f"revenue_mean_lag_{revision_window}"].astype(float)
    panel[f"revenue_estimate_revision_{revision_window}d"] = safe_div(
        rev_mean - rev_mean_lag,
        rev_mean_lag.abs(),
        require_positive_denom=True,
    )

    output = panel[["instrument_id", "date", *feature_names]].copy()
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
    "compute_estimate_features",
]
