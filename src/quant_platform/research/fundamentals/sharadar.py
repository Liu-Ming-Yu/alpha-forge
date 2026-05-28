"""Sharadar SF1 → research panel loader.

Loads the parquet cache produced by ``scripts/pull_sharadar_sf1.py`` and joins
its ticker column against the map produced by
``scripts/build_sharadar_ticker_map.py``, returning a long-format panel keyed
by ``(instrument_id, datekey)`` that is safe to use as the input to point-in-
time feature computation.

Point-in-time discipline
------------------------

* ``datekey`` is the SEC filing date — the moment the row became publicly
  knowable. **Always join feature frames into the bar/return panel on
  ``datekey`` (lagged by at least one trading day), never on
  ``calendardate``.** ``calendardate`` is the fiscal period end, which the
  market does not learn for ~30-90 days; joining on it leaks the future.
* The loader drops rows with a missing ``datekey`` and de-duplicates on
  ``(instrument_id, datekey)``, keeping the row with the latest
  ``lastupdated`` when Sharadar publishes a restated record under the same
  datekey.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAP_PATH = PROJECT_ROOT / "infra" / "config" / "sharadar_ticker_map.json"
DEFAULT_SF1_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "research"
    / "fundamentals"
    / "sharadar_sf1"
    / "sf1_arq.parquet"
)

# Required output columns. Callers can extend the projection but these always
# come back so downstream feature code can rely on them.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "ticker",
    "datekey",
    "calendardate",
)


@dataclass(frozen=True)
class SharadarPanel:
    """Result wrapper for a loaded Sharadar SF1 panel.

    Attributes
    ----------
    frame:
        Long-format DataFrame keyed by ``(instrument_id, datekey)``.
    instrument_coverage:
        Count of distinct instruments present in ``frame``.
    datekey_min, datekey_max:
        Date span of the loaded panel.
    dropped_no_instrument_id:
        Tickers in the parquet that did not appear in the ticker map.
    dropped_missing_datekey:
        Row count discarded for missing ``datekey``.
    duplicates_resolved:
        Row count discarded by the ``(instrument_id, datekey)`` de-dup pass.
    """

    frame: pd.DataFrame
    instrument_coverage: int
    datekey_min: pd.Timestamp | None
    datekey_max: pd.Timestamp | None
    dropped_no_instrument_id: tuple[str, ...]
    dropped_missing_datekey: int
    duplicates_resolved: int


def _load_ticker_to_instrument(map_path: Path) -> dict[str, str]:
    """Read the ticker map and invert it to ``sharadar_ticker -> instrument_id``."""
    payload: dict[str, Any] = json.loads(map_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = payload.get("mapping", {})
    out: dict[str, str] = {}
    for instrument_id, row in mapping.items():
        ticker = row.get("sharadar_ticker") or row.get("symbol")
        if not ticker:
            continue
        out[ticker.upper()] = instrument_id
    return out


def load_sector_map(map_path: Path = DEFAULT_MAP_PATH) -> dict[str, str]:
    """Return an ``instrument_id -> sector`` map from the Sharadar ticker map.

    Sectors come from Sharadar's own taxonomy (11 buckets across the 337-name
    universe: Technology, Financial Services, Healthcare, Industrials,
    Consumer Cyclical, Consumer Defensive, Basic Materials, Utilities, Real
    Estate, Energy, Communication Services). Used by the feature layer to
    compute sector-relative versions of the starter features.
    """
    payload: dict[str, Any] = json.loads(map_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = payload.get("mapping", {})
    return {
        instrument_id: str(row.get("sector"))
        for instrument_id, row in mapping.items()
        if row.get("sector")
    }


def load_sharadar_sf1_panel(
    *,
    parquet_path: Path = DEFAULT_SF1_PARQUET,
    map_path: Path = DEFAULT_MAP_PATH,
    columns: tuple[str, ...] | None = None,
) -> SharadarPanel:
    """Load Sharadar SF1 parquet, attach ``instrument_id``, de-duplicate.

    Parameters
    ----------
    parquet_path:
        Path to the SF1 parquet (defaults to the ARQ cache).
    map_path:
        Path to the ticker map JSON (defaults to
        ``infra/config/sharadar_ticker_map.json``).
    columns:
        Optional projection — restrict the returned frame to these columns
        plus the required key columns. ``None`` keeps every column in the
        parquet.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Sharadar SF1 parquet not found: {parquet_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Sharadar ticker map not found: {map_path}")

    df = pd.read_parquet(parquet_path)
    ticker_to_instrument = _load_ticker_to_instrument(map_path)

    # Normalize types — pandas reads parquet date columns as object; coerce.
    if "datekey" in df.columns:
        df["datekey"] = pd.to_datetime(df["datekey"], errors="coerce")
    if "calendardate" in df.columns:
        df["calendardate"] = pd.to_datetime(df["calendardate"], errors="coerce")
    if "lastupdated" in df.columns:
        df["lastupdated"] = pd.to_datetime(df["lastupdated"], errors="coerce")

    # Attach instrument_id via the map; record tickers that fall through.
    df["ticker_upper"] = df["ticker"].astype(str).str.upper()
    df["instrument_id"] = df["ticker_upper"].map(ticker_to_instrument)
    dropped_no_inst = tuple(
        sorted(set(df.loc[df["instrument_id"].isna(), "ticker_upper"].unique()))
    )
    df = df.dropna(subset=["instrument_id"]).copy()
    df.drop(columns=["ticker_upper"], inplace=True)

    # Drop rows with no datekey — they cannot be PIT-joined.
    before = len(df)
    df = df.dropna(subset=["datekey"]).copy()
    dropped_missing_datekey = before - len(df)

    # Resolve duplicates on (instrument_id, datekey) by keeping the row with
    # the most recent lastupdated — Sharadar occasionally republishes a row
    # under the same datekey when restating an originally-filed quarter.
    before = len(df)
    if "lastupdated" in df.columns:
        df = df.sort_values(["instrument_id", "datekey", "lastupdated"], na_position="first")
    else:
        df = df.sort_values(["instrument_id", "datekey"])
    df = df.drop_duplicates(subset=["instrument_id", "datekey"], keep="last").copy()
    duplicates_resolved = before - len(df)

    if columns is not None:
        keep = list(dict.fromkeys((*REQUIRED_COLUMNS, *columns)))
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"requested columns not present in panel: {missing}")
        df = df[keep].copy()

    # Stable column order: keys first, everything else after.
    front = list(REQUIRED_COLUMNS)
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].reset_index(drop=True)

    return SharadarPanel(
        frame=df,
        instrument_coverage=int(df["instrument_id"].nunique()),
        datekey_min=df["datekey"].min() if not df.empty else None,
        datekey_max=df["datekey"].max() if not df.empty else None,
        dropped_no_instrument_id=dropped_no_inst,
        dropped_missing_datekey=int(dropped_missing_datekey),
        duplicates_resolved=int(duplicates_resolved),
    )


__all__ = [
    "REQUIRED_COLUMNS",
    "SharadarPanel",
    "load_sector_map",
    "load_sharadar_sf1_panel",
]
