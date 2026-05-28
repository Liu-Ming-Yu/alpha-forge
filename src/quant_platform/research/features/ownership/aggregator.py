"""Per-record → per-(instrument, date) panel for ``ownership-v1``.

Three input streams feed the panel:

* :class:`Holding13FRecord` lines (institutional holdings) — collapse
  to per-(instrument, period_end) aggregates: total shares,
  unique-holder count.
* :class:`ShortInterestRecord` snapshots — already per-instrument; just
  forward-fill into the daily grid.
* :class:`SharesOutstandingRecord` snapshots — used to normalise the
  share counts into percentages of float.

Each input carries an ``available_at`` field (with sensible defaults
in the schema). The aggregator masks rows from dates earlier than
``available_at`` so the panel is point-in-time-safe.

Output frame columns:

* ``instrument_id``, ``date``
* ``institutional_shares_total`` — sum of all 13F holdings as-of the
  most recent available filing period.
* ``institutional_holder_count`` — count of distinct filers.
* ``short_interest_shares`` — most recent available short-interest.
* ``avg_daily_volume_shares`` — companion to short interest (for
  days-to-cover).
* ``shares_outstanding`` — most recent available shares outstanding.

The aggregator does not compute features — that's
``compute_ownership_features``. This module produces the rectangular
panel the feature compute reads from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.ownership.schemas import (
        Holding13FRecord,
        SharesOutstandingRecord,
        ShortInterestRecord,
    )


@dataclass(frozen=True)
class AggregatedOwnershipPanel:
    """Wide-format intermediate the feature compute reads.

    Keyed by ``(instrument_id, date)``. ``n_records_processed`` is the
    total count of input records (13F + short-interest + shares-out)
    seen; useful for coverage diagnostics.
    """

    frame: pd.DataFrame
    n_holdings_processed: int
    n_short_interest_processed: int
    n_shares_outstanding_processed: int


def _normalise_date(value: object) -> pd.Timestamp:
    """Convert any tz-aware datetime-like into a naive UTC
    midnight ``pd.Timestamp`` — matches the rest of the panel
    convention (naive datetime64[ns])."""
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def build_ownership_panel(
    *,
    holdings: Iterable[Holding13FRecord],
    short_interest: Iterable[ShortInterestRecord],
    shares_outstanding: Iterable[SharesOutstandingRecord],
    trading_dates: pd.DatetimeIndex,
    holding_13f_availability_lag_days: int = 45,
    short_interest_availability_lag_days: int = 8,
) -> AggregatedOwnershipPanel:
    """Materialise the per-(instrument, date) ownership panel.

    Parameters
    ----------
    holdings:
        Iterable of :class:`Holding13FRecord`.
    short_interest:
        Iterable of :class:`ShortInterestRecord`.
    shares_outstanding:
        Iterable of :class:`SharesOutstandingRecord`.
    trading_dates:
        The dates the panel must materialise rows for. Required —
        without a calendar the panel grain is undefined.
    holding_13f_availability_lag_days:
        Default lag from ``period_end`` to availability when a 13F
        record doesn't carry an explicit ``available_at``.
    short_interest_availability_lag_days:
        Same for short-interest records.
    """
    holdings_list = list(holdings)
    si_list = list(short_interest)
    so_list = list(shares_outstanding)

    # ------------------------------------------------------------------
    # 13F: collapse to per-(instrument, period_end) totals.
    # ------------------------------------------------------------------
    if holdings_list:
        holdings_df = pd.DataFrame(
            [
                {
                    "instrument_id": h.instrument_id,
                    "period_end": _normalise_date(h.period_end),
                    "available_at": _normalise_date(
                        h.available_at
                        if h.available_at is not None
                        else h.period_end + timedelta(days=holding_13f_availability_lag_days)
                    ),
                    "filer_id": h.filer_id,
                    "shares_held": h.shares_held,
                }
                for h in holdings_list
            ]
        )
        # One row per (instrument, period_end): sum shares + count
        # unique filers. ``available_at`` is the max across filers for
        # that period — institutions file independently, so the
        # earliest the entire panel-cell is comparable is when the
        # latest filer reports.
        holdings_panel = (
            holdings_df.groupby(["instrument_id", "period_end"], sort=False, as_index=False)
            .agg(
                institutional_shares_total=("shares_held", "sum"),
                institutional_holder_count=("filer_id", "nunique"),
                available_at=("available_at", "max"),
            )
            .sort_values(["instrument_id", "period_end"])
            .reset_index(drop=True)
        )
    else:
        holdings_panel = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "institutional_shares_total": pd.Series(dtype="int64"),
                "institutional_holder_count": pd.Series(dtype="int64"),
                "available_at": pd.Series(dtype="datetime64[ns]"),
            }
        )

    # ------------------------------------------------------------------
    # Short interest: one row per (instrument, settlement_date).
    # ------------------------------------------------------------------
    if si_list:
        si_panel = (
            pd.DataFrame(
                [
                    {
                        "instrument_id": s.instrument_id,
                        "settlement_date": _normalise_date(s.settlement_date),
                        "available_at": _normalise_date(
                            s.available_at
                            if s.available_at is not None
                            else s.settlement_date
                            + timedelta(days=short_interest_availability_lag_days)
                        ),
                        "short_interest_shares": s.short_interest_shares,
                        "avg_daily_volume_shares": s.avg_daily_volume_shares,
                    }
                    for s in si_list
                ]
            )
            .sort_values(["instrument_id", "settlement_date"])
            .reset_index(drop=True)
        )
    else:
        si_panel = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "settlement_date": pd.Series(dtype="datetime64[ns]"),
                "available_at": pd.Series(dtype="datetime64[ns]"),
                "short_interest_shares": pd.Series(dtype="int64"),
                "avg_daily_volume_shares": pd.Series(dtype="float64"),
            }
        )

    # ------------------------------------------------------------------
    # Shares outstanding: one row per (instrument, period_end). Assumed
    # available immediately (shares-outstanding is in fundamentals data
    # which the platform already handles with its own PIT semantics).
    # ------------------------------------------------------------------
    if so_list:
        so_panel = (
            pd.DataFrame(
                [
                    {
                        "instrument_id": r.instrument_id,
                        "period_end": _normalise_date(r.period_end),
                        "shares_outstanding": r.shares_outstanding,
                    }
                    for r in so_list
                ]
            )
            .sort_values(["instrument_id", "period_end"])
            .reset_index(drop=True)
        )
    else:
        so_panel = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "shares_outstanding": pd.Series(dtype="int64"),
            }
        )

    # ------------------------------------------------------------------
    # Materialise the (instrument, date) grid and forward-fill the
    # most recent available record for each stream.
    # ------------------------------------------------------------------
    instruments = sorted(
        set(holdings_panel["instrument_id"])
        | set(si_panel["instrument_id"])
        | set(so_panel["instrument_id"])
    )
    if not instruments or len(trading_dates) == 0:
        empty = _empty_output_frame()
        return AggregatedOwnershipPanel(
            frame=empty,
            n_holdings_processed=len(holdings_list),
            n_short_interest_processed=len(si_list),
            n_shares_outstanding_processed=len(so_list),
        )

    trading_dates_naive = pd.DatetimeIndex(
        [_normalise_date(d) for d in trading_dates]
    ).sort_values()
    grid = pd.MultiIndex.from_product(
        [instruments, trading_dates_naive], names=["instrument_id", "date"]
    ).to_frame(index=False)

    grid = _join_pit_stream(
        grid,
        stream=holdings_panel,
        join_columns=("institutional_shares_total", "institutional_holder_count"),
        availability_column="available_at",
    )
    grid = _join_pit_stream(
        grid,
        stream=si_panel,
        join_columns=("short_interest_shares", "avg_daily_volume_shares"),
        availability_column="available_at",
    )
    grid = _join_pit_stream(
        grid,
        stream=so_panel,
        join_columns=("shares_outstanding",),
        availability_column="period_end",  # assume shares-out is immediately PIT
    )

    return AggregatedOwnershipPanel(
        frame=grid,
        n_holdings_processed=len(holdings_list),
        n_short_interest_processed=len(si_list),
        n_shares_outstanding_processed=len(so_list),
    )


def _join_pit_stream(
    grid: pd.DataFrame,
    *,
    stream: pd.DataFrame,
    join_columns: tuple[str, ...],
    availability_column: str,
) -> pd.DataFrame:
    """As-of join a PIT-respecting stream onto the per-day grid.

    For each (instrument, date) row in the grid, attach the values
    from the stream's most recent record whose ``availability_column``
    is ``<= date``. Rows with no eligible stream record stay NaN.

    Uses :func:`pandas.merge_asof` per-instrument so we don't have to
    custom-iterate.
    """
    if stream.empty:
        # Use np.nan (float-compatible) rather than pd.NA so downstream
        # ``.astype(float)`` calls in the features module don't choke on
        # the NAType.
        import numpy as np

        for col in join_columns:
            grid[col] = np.nan
        return grid

    # ``merge_asof`` requires both sides globally sorted by the asof-key
    # (date / availability_column). The ``by="instrument_id"`` argument
    # scopes the asof matching per-group, but the underlying sort must
    # be on the asof-key alone — sorting by ["instrument_id", "date"]
    # would leave ``date`` non-monotonic across instrument boundaries
    # and trigger the "left keys must be sorted" check.
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
    # Restore the canonical (instrument_id, date) sort order so callers
    # see the panel in the documented layout, and drop the asof-key
    # from the stream (we don't surface it on the output frame — the
    # grid's ``date`` is authoritative).
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
            "institutional_shares_total": pd.Series(dtype="float64"),
            "institutional_holder_count": pd.Series(dtype="float64"),
            "short_interest_shares": pd.Series(dtype="float64"),
            "avg_daily_volume_shares": pd.Series(dtype="float64"),
            "shares_outstanding": pd.Series(dtype="float64"),
        }
    )


__all__ = [
    "AggregatedOwnershipPanel",
    "build_ownership_panel",
]
