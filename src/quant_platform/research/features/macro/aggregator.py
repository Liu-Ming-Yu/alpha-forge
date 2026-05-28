"""Per-series → per-date panel for ``macro-v1``.

Unlike the other families, macro features are **scalar per date**:
the 10-year Treasury yield is one number per day, not one number per
(instrument, day). The aggregator produces a single per-date frame
with one column per FRED series ID, forward-filled across the
trading calendar via ``pd.merge_asof(direction="backward")``.

The downstream feature compute broadcasts these per-date values
across the operator-supplied instrument list to produce a standard
(instrument_id, date)-keyed FeatureFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.macro.schemas import MacroSeriesValue


@dataclass(frozen=True)
class AggregatedMacroPanel:
    """Per-date macro panel.

    The frame is keyed by ``date`` (single index column) — it doesn't
    carry ``instrument_id``. The compute layer broadcasts the
    per-date values across instruments to produce the
    (instrument, date) FeatureFrame.
    """

    frame: pd.DataFrame
    n_observations_processed: int


def _normalise_date(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def build_macro_panel(
    *,
    series_values: Iterable[MacroSeriesValue],
    trading_dates: pd.DatetimeIndex,
    required_series_ids: tuple[str, ...],
) -> AggregatedMacroPanel:
    """Materialise the per-date macro panel from raw observations.

    Parameters
    ----------
    series_values:
        Iterable of :class:`MacroSeriesValue`. Records whose
        ``series_id`` is not in ``required_series_ids`` are silently
        ignored — operators can pass a superset of series safely.
    trading_dates:
        Calendar of dates the panel materialises rows for.
    required_series_ids:
        FRED series IDs the family needs. Each becomes a column in
        the output frame. Missing-from-input series get an all-NaN
        column rather than failing — keeps the family resilient to
        partial feeds while the downstream feature compute decides
        what to do with the missing data.
    """
    values_list = list(series_values)
    required = set(required_series_ids)

    if len(trading_dates) == 0:
        return AggregatedMacroPanel(
            frame=_empty_output_frame(required_series_ids),
            n_observations_processed=len(values_list),
        )

    trading_dates_naive = pd.DatetimeIndex(
        [_normalise_date(d) for d in trading_dates]
    ).sort_values()
    grid = pd.DataFrame({"date": trading_dates_naive})

    if not values_list:
        for sid in required_series_ids:
            grid[sid] = np.nan
        return AggregatedMacroPanel(frame=grid, n_observations_processed=0)

    # Filter to the required series; group by series; build per-series
    # streams ready for per-series merge_asof.
    raw = pd.DataFrame(
        [
            {
                "series_id": v.series_id,
                "observation_date": _normalise_date(v.observation_date),
                "value": float(v.value),
            }
            for v in values_list
            if v.series_id in required
        ]
    )

    # Each required series gets a column. Series with no observations
    # get an all-NaN column.
    for sid in required_series_ids:
        series_stream = raw[raw["series_id"] == sid][["observation_date", "value"]]
        if series_stream.empty:
            grid[sid] = np.nan
            continue
        series_stream = (
            series_stream.rename(columns={"value": sid})
            .sort_values("observation_date", kind="stable")
            .reset_index(drop=True)
        )
        grid = pd.merge_asof(
            grid.sort_values("date", kind="stable").reset_index(drop=True),
            series_stream,
            left_on="date",
            right_on="observation_date",
            direction="backward",
        )
        grid = grid.drop(columns=["observation_date"])

    return AggregatedMacroPanel(
        frame=grid.sort_values("date").reset_index(drop=True),
        n_observations_processed=len(values_list),
    )


def _empty_output_frame(required_series_ids: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            **{sid: pd.Series(dtype="float64") for sid in required_series_ids},
        }
    )


__all__ = ["AggregatedMacroPanel", "build_macro_panel"]
