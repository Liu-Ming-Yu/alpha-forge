"""Per-record → per-(instrument, date) panel for ``estimates-v1``.

Two input streams feed the panel:

* :class:`ConsensusSnapshot` — daily consensus snapshots for one
  ``(instrument, target_period, estimate_kind)`` triple. The
  aggregator filters by target period + kind, then forward-fills onto
  the daily grid.
* :class:`EarningsSurpriseRecord` — historical earnings-actual vs
  consensus events. The aggregator computes the mean % surprise over
  the trailing ``surprise_lookback_quarters`` events.

PIT safety:

* Consensus snapshots are forward-filled by their ``snapshot_date``
  (the snapshot's own date is when the consensus was known publicly).
* Surprise records are masked from the panel until
  ``reported_at <= panel_date``.

Output frame columns (one row per (instrument, date)):

* ``instrument_id``, ``date``
* ``eps_mean``, ``eps_std``, ``eps_n``,
  ``eps_n_up_30d``, ``eps_n_down_30d`` — current consensus + revision counts.
* ``eps_mean_lag<window>`` — consensus from <window> calendar days ago,
  for the revision feature.
* ``revenue_mean``, ``revenue_mean_lag<window>`` — same for revenue.
* ``eps_surprise_mean_recent`` — mean % surprise over the last
  ``surprise_lookback_quarters`` reported quarters.

The aggregator does not compute features — that's
``compute_estimate_features``. This module produces the rectangular
panel the feature compute reads from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.estimates.schemas import (
        ConsensusSnapshot,
        EarningsSurpriseRecord,
    )


@dataclass(frozen=True)
class AggregatedEstimatesPanel:
    """Wide-format intermediate the feature compute reads.

    Keyed by ``(instrument_id, date)``. ``n_*_processed`` counts are
    the total records consumed per stream — useful for coverage
    diagnostics.
    """

    frame: pd.DataFrame
    n_consensus_processed: int
    n_surprise_processed: int


def _normalise_date(value: object) -> pd.Timestamp:
    """Convert any tz-aware datetime-like into a naive UTC midnight
    ``pd.Timestamp`` — matches the rest of the panel convention."""
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def build_estimates_panel(
    *,
    consensus_snapshots: Iterable[ConsensusSnapshot],
    surprise_records: Iterable[EarningsSurpriseRecord],
    trading_dates: pd.DatetimeIndex,
    eps_target_period: str,
    revenue_target_period: str,
    revision_window_days: int,
    surprise_lookback_quarters: int,
) -> AggregatedEstimatesPanel:
    """Materialise the per-(instrument, date) estimates panel.

    Parameters
    ----------
    consensus_snapshots:
        Iterable of :class:`ConsensusSnapshot`. Records whose
        ``target_period`` / ``estimate_kind`` doesn't match the
        configured EPS / revenue target are silently filtered out.
    surprise_records:
        Iterable of :class:`EarningsSurpriseRecord`. Masked by
        ``reported_at <= panel_date`` so the surprise feature is
        PIT-safe.
    trading_dates:
        The dates the panel materialises rows for. Required.
    eps_target_period, revenue_target_period:
        Which fiscal period the consensus snapshots are filtered to
        (e.g. ``"FY1"``).
    revision_window_days:
        Calendar-day lookback for the lagged-consensus columns
        (``eps_mean_lag_<N>``, ``revenue_mean_lag_<N>``).
    surprise_lookback_quarters:
        Number of trailing fiscal-period surprises to average.
    """
    consensus_list = list(consensus_snapshots)
    surprise_list = list(surprise_records)

    eps_panel = _consensus_kind_panel(
        consensus_list,
        target_period=eps_target_period,
        estimate_kind="eps",
        revision_window_days=revision_window_days,
        column_prefix="eps_",
    )
    revenue_panel = _consensus_kind_panel(
        consensus_list,
        target_period=revenue_target_period,
        estimate_kind="revenue",
        revision_window_days=revision_window_days,
        column_prefix="revenue_",
    )
    surprise_panel = _surprise_panel(
        surprise_list, surprise_lookback_quarters=surprise_lookback_quarters
    )

    # ------------------------------------------------------------------
    # Materialise the (instrument, date) grid.
    # ------------------------------------------------------------------
    instruments = sorted(
        set(eps_panel["instrument_id"])
        | set(revenue_panel["instrument_id"])
        | set(surprise_panel["instrument_id"])
    )
    if not instruments or len(trading_dates) == 0:
        return AggregatedEstimatesPanel(
            frame=_empty_output_frame(revision_window_days),
            n_consensus_processed=len(consensus_list),
            n_surprise_processed=len(surprise_list),
        )

    trading_dates_naive = pd.DatetimeIndex(
        [_normalise_date(d) for d in trading_dates]
    ).sort_values()
    grid = pd.MultiIndex.from_product(
        [instruments, trading_dates_naive], names=["instrument_id", "date"]
    ).to_frame(index=False)

    eps_join_cols = (
        "eps_mean",
        "eps_std",
        "eps_n",
        "eps_n_up_30d",
        "eps_n_down_30d",
    )
    grid = _join_pit_stream(
        grid,
        stream=eps_panel,
        availability_column="snapshot_date",
        join_columns=eps_join_cols,
    )
    # Lagged EPS mean: same join logic, but against
    # ``snapshot_date + revision_window_days`` so the row at panel
    # date T pulls the consensus from T - window. Compute by adding
    # the window to the stream's snapshot_date column and re-joining.
    eps_lag_stream = eps_panel.copy()
    eps_lag_stream["snapshot_date_plus_window"] = eps_lag_stream["snapshot_date"] + pd.Timedelta(
        days=revision_window_days
    )
    eps_lag_stream = eps_lag_stream.rename(
        columns={"eps_mean": f"eps_mean_lag_{revision_window_days}"}
    )
    grid = _join_pit_stream(
        grid,
        stream=eps_lag_stream,
        availability_column="snapshot_date_plus_window",
        join_columns=(f"eps_mean_lag_{revision_window_days}",),
    )

    revenue_join_cols = (
        "revenue_mean",
        "revenue_n",
    )
    grid = _join_pit_stream(
        grid,
        stream=revenue_panel,
        availability_column="snapshot_date",
        join_columns=revenue_join_cols,
    )
    revenue_lag_stream = revenue_panel.copy()
    revenue_lag_stream["snapshot_date_plus_window"] = revenue_lag_stream[
        "snapshot_date"
    ] + pd.Timedelta(days=revision_window_days)
    revenue_lag_stream = revenue_lag_stream.rename(
        columns={"revenue_mean": f"revenue_mean_lag_{revision_window_days}"}
    )
    grid = _join_pit_stream(
        grid,
        stream=revenue_lag_stream,
        availability_column="snapshot_date_plus_window",
        join_columns=(f"revenue_mean_lag_{revision_window_days}",),
    )

    grid = _join_pit_stream(
        grid,
        stream=surprise_panel,
        availability_column="reported_at",
        join_columns=("eps_surprise_mean_recent",),
    )

    return AggregatedEstimatesPanel(
        frame=grid,
        n_consensus_processed=len(consensus_list),
        n_surprise_processed=len(surprise_list),
    )


def _consensus_kind_panel(
    consensus_list: list[ConsensusSnapshot],
    *,
    target_period: str,
    estimate_kind: str,
    revision_window_days: int,
    column_prefix: str,
) -> pd.DataFrame:
    """Filter consensus snapshots to one (target_period, kind) and
    return a long-format stream ready for as-of joining.

    The shape of the returned frame depends on the kind: EPS carries
    revision counts and std; revenue currently only carries mean and
    n (which is all v1's revenue feature needs)."""
    del revision_window_days  # currently unused — lagged-mean is joined separately
    matching = [
        s
        for s in consensus_list
        if s.target_period == target_period and s.estimate_kind == estimate_kind
    ]
    if not matching:
        if estimate_kind == "eps":
            return pd.DataFrame(
                {
                    "instrument_id": pd.Series(dtype=str),
                    "snapshot_date": pd.Series(dtype="datetime64[ns]"),
                    "eps_mean": pd.Series(dtype="float64"),
                    "eps_std": pd.Series(dtype="float64"),
                    "eps_n": pd.Series(dtype="float64"),
                    "eps_n_up_30d": pd.Series(dtype="float64"),
                    "eps_n_down_30d": pd.Series(dtype="float64"),
                }
            )
        return pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "snapshot_date": pd.Series(dtype="datetime64[ns]"),
                "revenue_mean": pd.Series(dtype="float64"),
                "revenue_n": pd.Series(dtype="float64"),
            }
        )

    rows: list[dict[str, object]] = []
    for s in matching:
        row: dict[str, object] = {
            "instrument_id": s.instrument_id,
            "snapshot_date": _normalise_date(s.snapshot_date),
            f"{column_prefix}mean": float(s.mean_estimate),
            f"{column_prefix}n": float(s.n_estimates),
        }
        if estimate_kind == "eps":
            row["eps_std"] = float(s.std_estimate) if s.std_estimate is not None else np.nan
            row["eps_n_up_30d"] = float(s.n_up_revisions_30d)
            row["eps_n_down_30d"] = float(s.n_down_revisions_30d)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["instrument_id", "snapshot_date"]).reset_index(drop=True)


def _surprise_panel(
    surprise_list: list[EarningsSurpriseRecord],
    *,
    surprise_lookback_quarters: int,
) -> pd.DataFrame:
    """For each surprise record, compute the trailing-N-quarter mean
    % surprise visible AS-OF its ``reported_at`` date.

    The returned stream has one row per surprise event, with the
    ``eps_surprise_mean_recent`` value being the trailing mean
    INCLUDING that event. The as-of join then forward-fills this
    value across subsequent panel dates.
    """
    if not surprise_list:
        return pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "reported_at": pd.Series(dtype="datetime64[ns]"),
                "eps_surprise_mean_recent": pd.Series(dtype="float64"),
            }
        )

    rows = [
        {
            "instrument_id": r.instrument_id,
            "reported_at": _normalise_date(r.reported_at),
            "actual_eps": float(r.actual_eps),
            "consensus_mean_eps": float(r.consensus_mean_eps),
        }
        for r in surprise_list
    ]
    df = pd.DataFrame(rows).sort_values(["instrument_id", "reported_at"]).reset_index(drop=True)
    # % surprise per event: (actual - consensus) / |consensus|. NaN
    # when consensus is exactly zero.
    denom = df["consensus_mean_eps"].abs()
    pct_surprise = (df["actual_eps"] - df["consensus_mean_eps"]) / denom.where(denom > 0, np.nan)
    df["pct_surprise"] = pct_surprise
    # Trailing-N mean per instrument, including the current event.
    df["eps_surprise_mean_recent"] = (
        df.groupby("instrument_id")["pct_surprise"]
        .rolling(window=surprise_lookback_quarters, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df[["instrument_id", "reported_at", "eps_surprise_mean_recent"]]


def _join_pit_stream(
    grid: pd.DataFrame,
    *,
    stream: pd.DataFrame,
    availability_column: str,
    join_columns: tuple[str, ...],
) -> pd.DataFrame:
    """As-of join a PIT-respecting stream onto the per-day grid.

    Same pattern as ``ownership/aggregator._join_pit_stream``: the
    left frame is globally sorted by date, the stream by its
    availability key, then ``merge_asof(direction="backward",
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


def _empty_output_frame(revision_window_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "date": pd.Series(dtype="datetime64[ns]"),
            "eps_mean": pd.Series(dtype="float64"),
            "eps_std": pd.Series(dtype="float64"),
            "eps_n": pd.Series(dtype="float64"),
            "eps_n_up_30d": pd.Series(dtype="float64"),
            "eps_n_down_30d": pd.Series(dtype="float64"),
            f"eps_mean_lag_{revision_window_days}": pd.Series(dtype="float64"),
            "revenue_mean": pd.Series(dtype="float64"),
            "revenue_n": pd.Series(dtype="float64"),
            f"revenue_mean_lag_{revision_window_days}": pd.Series(dtype="float64"),
            "eps_surprise_mean_recent": pd.Series(dtype="float64"),
        }
    )


# Suppress an unused-import warning when timedelta is only referenced
# in module-level docs; pandas.Timedelta is what we actually use.
_ = timedelta


__all__ = [
    "AggregatedEstimatesPanel",
    "build_estimates_panel",
]
