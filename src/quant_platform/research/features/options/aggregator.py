"""Per-record → per-(instrument, date) panel for ``options-v1``.

The family takes one :class:`OptionsSnapshot` stream as input. The
aggregator's only job is to forward-fill the most-recent snapshot
onto a daily trading-date grid via ``pd.merge_asof`` — same pattern
as the ownership / estimates aggregators.

PIT safety: a snapshot's ``snapshot_date`` is when its values are
publicly known (vendors publish derived options metrics intraday or
end-of-day; we treat the snapshot date itself as the availability
date). Operators with strict end-of-day-only feeds should ingest
records with ``snapshot_date`` = the date the data is FIRST
available, not the trading date the surface was fit against.

Output frame columns (one row per (instrument, date)):

* ``instrument_id``, ``date``
* ``iv_30d_atm``, ``iv_60d_atm`` (or whatever tenors the config picks)
* ``iv_25d_call``, ``iv_25d_put``
* ``put_volume``, ``call_volume``
* ``put_open_interest``, ``call_open_interest``
* ``realized_vol_21d`` (or the configured window)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.options.schemas import OptionsSnapshot


@dataclass(frozen=True)
class AggregatedOptionsPanel:
    """Wide-format intermediate the feature compute reads."""

    frame: pd.DataFrame
    n_snapshots_processed: int


def _normalise_date(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def build_options_panel(
    *,
    snapshots: Iterable[OptionsSnapshot],
    trading_dates: pd.DatetimeIndex,
) -> AggregatedOptionsPanel:
    """Materialise the per-(instrument, date) options panel via
    forward-fill from snapshots.
    """
    snapshots_list = list(snapshots)

    if not snapshots_list or len(trading_dates) == 0:
        return AggregatedOptionsPanel(
            frame=_empty_output_frame(), n_snapshots_processed=len(snapshots_list)
        )

    raw = (
        pd.DataFrame(
            [
                {
                    "instrument_id": s.instrument_id,
                    "snapshot_date": _normalise_date(s.snapshot_date),
                    "iv_30d_atm": np.nan if s.iv_30d_atm is None else float(s.iv_30d_atm),
                    "iv_60d_atm": np.nan if s.iv_60d_atm is None else float(s.iv_60d_atm),
                    "iv_25d_call": np.nan if s.iv_25d_call is None else float(s.iv_25d_call),
                    "iv_25d_put": np.nan if s.iv_25d_put is None else float(s.iv_25d_put),
                    "put_volume": float(s.put_volume),
                    "call_volume": float(s.call_volume),
                    "put_open_interest": float(s.put_open_interest),
                    "call_open_interest": float(s.call_open_interest),
                    "realized_vol_21d": (
                        np.nan if s.realized_vol_21d is None else float(s.realized_vol_21d)
                    ),
                }
                for s in snapshots_list
            ]
        )
        .sort_values(["instrument_id", "snapshot_date"])
        .reset_index(drop=True)
    )

    instruments = sorted(raw["instrument_id"].unique())
    trading_dates_naive = pd.DatetimeIndex(
        [_normalise_date(d) for d in trading_dates]
    ).sort_values()
    grid = pd.MultiIndex.from_product(
        [instruments, trading_dates_naive], names=["instrument_id", "date"]
    ).to_frame(index=False)

    join_cols = (
        "iv_30d_atm",
        "iv_60d_atm",
        "iv_25d_call",
        "iv_25d_put",
        "put_volume",
        "call_volume",
        "put_open_interest",
        "call_open_interest",
        "realized_vol_21d",
    )
    grid = _join_pit_stream(
        grid,
        stream=raw,
        availability_column="snapshot_date",
        join_columns=join_cols,
    )

    return AggregatedOptionsPanel(frame=grid, n_snapshots_processed=len(snapshots_list))


def _join_pit_stream(
    grid: pd.DataFrame,
    *,
    stream: pd.DataFrame,
    availability_column: str,
    join_columns: tuple[str, ...],
) -> pd.DataFrame:
    """As-of join a PIT-respecting stream onto the per-day grid.

    Same pattern as ownership/estimates aggregators: sort both sides
    by the asof-key globally, ``merge_asof(direction="backward",
    by="instrument_id")`` does the per-instrument forward-fill.
    """
    if stream.empty:
        for col in join_columns:
            grid[col] = np.nan
        return grid

    grid_sorted = grid.sort_values("date", kind="stable").reset_index(drop=True)
    stream_sorted = stream.sort_values(availability_column, kind="stable").reset_index(drop=True)

    joined = pd.merge_asof(
        grid_sorted,
        stream_sorted[
            [
                "instrument_id",
                availability_column,
                *join_columns,
            ]
        ],
        left_on="date",
        right_on=availability_column,
        by="instrument_id",
        direction="backward",
    )
    joined = (
        joined.drop(columns=[availability_column])
        .sort_values(["instrument_id", "date"])
        .reset_index(drop=True)
    )
    return joined


def _empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "date": pd.Series(dtype="datetime64[ns]"),
            "iv_30d_atm": pd.Series(dtype="float64"),
            "iv_60d_atm": pd.Series(dtype="float64"),
            "iv_25d_call": pd.Series(dtype="float64"),
            "iv_25d_put": pd.Series(dtype="float64"),
            "put_volume": pd.Series(dtype="float64"),
            "call_volume": pd.Series(dtype="float64"),
            "put_open_interest": pd.Series(dtype="float64"),
            "call_open_interest": pd.Series(dtype="float64"),
            "realized_vol_21d": pd.Series(dtype="float64"),
        }
    )


__all__ = ["AggregatedOptionsPanel", "build_options_panel"]
