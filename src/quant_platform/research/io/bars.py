"""Pandas-friendly bar reader for research scripts.

The canonical bar reader is :meth:`ParquetBarStore.get_bars`, which is async,
session-bound, returns ``MarketBar`` objects with ``Decimal`` fields, and --
critically -- filters on ``bar_seconds`` before returning rows. Research
scripts (walk-forwards, notebooks) typically want a ``pandas.DataFrame``
instead, and reach for ``pd.read_parquet`` directly. Doing so is unsafe:
in the pre-split layout (and during the migration window) yearly parquet
files for some instruments co-mingle daily (``bar_seconds=86400``) and
1-minute (``bar_seconds=60``) bars in the same file, so an unfiltered read
silently contaminates a daily-frequency series with intraday rows.
Computing a 21-step "daily" forward return on the resulting frame then
walks 21 minutes, not 21 days, and produces multi-million-percent returns.

Use :func:`load_daily_bars` for daily series and :func:`load_bars` with an
explicit ``bar_seconds`` for any other frequency. These functions:

* Read from the post-split on-disk layout
  ``{root}/bars/{instrument_id}/{daily|intraday}/{year}.parquet`` and
  transparently fall back to the legacy mixed file
  ``{root}/bars/{instrument_id}/{year}.parquet`` for years that have not
  been migrated yet.
* Always require ``bar_seconds`` (no default that could silently mix
  frequencies) and filter on it before returning, so even reading a legacy
  mixed file is safe.
* Optionally clip to a ``[start, end]`` UTC timestamp window.

They do NOT apply corporate-action adjustments -- only ``ParquetBarStore``
does. Use these for raw-bar diagnostics, not for production return series.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from datetime import datetime

DEFAULT_OBJECT_STORE_ROOT = Path("data/parquet")
DAILY_BAR_SECONDS = 86_400
DAILY_BUCKET = "daily"
INTRADAY_BUCKET = "intraday"


def _bucket_for(bar_seconds: int) -> str:
    return DAILY_BUCKET if bar_seconds == DAILY_BAR_SECONDS else INTRADAY_BUCKET


def load_bars(
    instrument_id: str | uuid.UUID,
    bar_seconds: int,
    *,
    root: str | Path = DEFAULT_OBJECT_STORE_ROOT,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load bars for one instrument at one frequency as a DataFrame.

    Reads the post-split bucketed file for the requested ``bar_seconds`` and,
    if a legacy mixed file still exists for the same ``(instrument, year)``,
    reads that too and filters on ``bar_seconds``. Callers can never
    accidentally mix frequencies regardless of which layout is on disk.

    Args:
        instrument_id: UUID of the instrument; coerced via ``str()``.
        bar_seconds: Required granularity to keep (e.g. ``86400`` for daily,
            ``60`` for 1-minute). Rows with any other ``bar_seconds`` are
            dropped.
        root: Object-store root containing the ``bars/`` subdirectory.
            Defaults to ``data/parquet``.
        start, end: Optional inclusive UTC timestamp window. Both ends are
            inclusive to match :meth:`ParquetBarStore.get_bars`. If supplied
            they also restrict which yearly files are read.
        columns: Optional list of columns to return. ``bar_seconds`` (and
            ``timestamp`` when a window is given) are always loaded from
            disk to enforce the filter, but they are not added to the
            returned frame unless the caller asks for them.

    Returns:
        DataFrame with one row per surviving bar, sorted by ``timestamp``.
        Index is a fresh ``RangeIndex``.
    """
    inst = str(instrument_id)
    bars_root = Path(root) / "bars" / inst
    bucket_dir = bars_root / _bucket_for(bar_seconds)

    years = _years_in_window(start, end, bars_root, bucket_dir)
    if not years:
        return _empty_frame(columns)

    needed: list[str] | None
    if columns is None:
        needed = None  # read everything
    else:
        # bar_seconds is mandatory to enforce the filter; timestamp is
        # mandatory whenever we have to apply a window.
        required = {"bar_seconds", *columns}
        if start is not None or end is not None:
            required.add("timestamp")
        needed = list(required)

    frames: list[pd.DataFrame] = []
    for year in years:
        for path in (bucket_dir / f"{year}.parquet", bars_root / f"{year}.parquet"):
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=needed)
            df = df[df["bar_seconds"] == bar_seconds]
            if start is not None:
                df = df[df["timestamp"] >= start]
            if end is not None:
                df = df[df["timestamp"] <= end]
            if not df.empty:
                frames.append(df)

    if not frames:
        return _empty_frame(columns)

    out = pd.concat(frames, ignore_index=True)
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)

    if columns is not None:
        # Drop the bar_seconds / timestamp columns we forced on disk if the
        # caller did not ask for them.
        keep = [c for c in columns if c in out.columns]
        out = out[keep]

    return out


def load_daily_bars(
    instrument_id: str | uuid.UUID,
    *,
    root: str | Path = DEFAULT_OBJECT_STORE_ROOT,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load daily (``bar_seconds=86400``) bars for one instrument.

    Thin wrapper around :func:`load_bars` that pins the frequency so research
    scripts cannot accidentally pick up the 1-minute rows that started
    appearing in 2025 yearly parquet files for a subset of instruments.
    """
    return load_bars(
        instrument_id,
        DAILY_BAR_SECONDS,
        root=root,
        start=start,
        end=end,
        columns=columns,
    )


def _years_in_window(
    start: datetime | None,
    end: datetime | None,
    bars_root: Path,
    bucket_dir: Path,
) -> list[int]:
    """Years to consider; honours the window if supplied else walks disk.

    Walks both the bucket subdirectory (post-split layout) and the
    instrument root (legacy mixed files) so a partially-migrated instrument
    is still readable.
    """
    if start is not None and end is not None:
        return list(range(start.year, end.year + 1))

    discovered: set[int] = set()
    for directory in (bucket_dir, bars_root):
        if not directory.exists():
            continue
        for child in directory.iterdir():
            if not child.is_file() or child.suffix != ".parquet":
                continue
            try:
                discovered.add(int(child.stem))
            except ValueError:
                continue

    years = sorted(discovered)
    if start is not None:
        years = [y for y in years if y >= start.year]
    if end is not None:
        years = [y for y in years if y <= end.year]
    return years


def _empty_frame(columns: Sequence[str] | None) -> pd.DataFrame:
    if columns is None:
        return pd.DataFrame()
    return pd.DataFrame(columns=list(columns))
