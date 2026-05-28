"""Bulk-pull ``SHARADAR/SF1`` for the configured universe and cache as parquet.

Reads the ticker map built by ``scripts/build_sharadar_ticker_map.py``, batches
its tickers, pulls every available ``SHARADAR/SF1`` row at the requested
dimension(s), and writes one parquet file per dimension to a research-side
cache directory.

The cache is **research-only** — no Postgres writes, no service-tier plumbing.
This keeps the first-cut fundamentals workstream cheap to kill if the
walk-forward gate refuses the resulting feature set.

Usage::

    python scripts/pull_sharadar_sf1.py \\
        --map infra/config/sharadar_ticker_map.json \\
        --out data/parquet/research/fundamentals/sharadar_sf1 \\
        --dimension ARQ

The Nasdaq Data Link key is read from the environment variable
``QP__DATA_INGEST__NASDAQ_DATA_LINK_API_KEY``. A project-root ``.env`` is
honored without ``python-dotenv``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = PROJECT_ROOT / "infra" / "config" / "sharadar_ticker_map.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "parquet" / "research" / "fundamentals" / "sharadar_sf1"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1"
ENV_KEY_NAME = "QP__DATA_INGEST__NASDAQ_DATA_LINK_API_KEY"

# SF1 has 112 columns; we pull a focused subset that covers the starter
# quality+value feature set plus controls. Adding columns later is cheap —
# the request is the only cost, the cache is rebuilt on each run.
SF1_COLUMNS = [
    # keys / metadata
    "ticker",
    "dimension",
    "calendardate",  # fiscal period end
    "datekey",  # date the filing became public (PIT — use this for joins!)
    "reportperiod",
    "fiscalperiod",
    "lastupdated",
    # income statement
    "revenue",
    "cor",
    "gp",
    "opex",
    "netinc",
    # balance sheet
    "assets",
    "liabilities",
    "equity",
    "equityavg",
    "debt",
    "cashneq",
    "intangibles",
    "sharesbas",
    # cash flow
    "ncfo",
    "ncfi",
    "ncff",
    "fcf",
    "capex",
    # market / valuation (Sharadar-computed)
    "marketcap",
    "ev",
    "pe",
    "pb",
    "ps",
    "divyield",
    # margins / returns (Sharadar-computed)
    "roe",
    "roa",
    "roic",
    "grossmargin",
    "netmargin",
    # `fcfmargin` does NOT exist in SF1 (probed; the API 403s with QEPx06).
    # Derive in the feature step as fcf / revenue when needed.
]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _http_get_json(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    # URL is built from API_BASE (a pinned https host) plus urlencoded params,
    # never user-controlled scheme — S310's file:/custom-scheme worry is moot.
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "quant-platform/pull-sharadar-sf1"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}")
        body = resp.read()
    return json.loads(body)


def _batch(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_sf1_for_tickers(
    api_key: str,
    tickers: list[str],
    dimension: str,
    *,
    batch_size: int = 50,
    page_pause_seconds: float = 0.3,
) -> pd.DataFrame:
    """Pull every SF1 row for ``tickers`` at ``dimension``.

    Batches ``tickers`` into groups of ``batch_size`` to keep the URL safely
    under any proxy length limit. Paginates each batch via ``next_cursor_id``.
    """
    columns_param = ",".join(SF1_COLUMNS)
    frames: list[pd.DataFrame] = []
    total_rows = 0
    batches = _batch(sorted(tickers), batch_size)
    for batch_idx, batch in enumerate(batches, start=1):
        ticker_param = ",".join(batch)
        base_params = {
            "ticker": ticker_param,
            "dimension": dimension,
            "qopts.columns": columns_param,
            "api_key": api_key,
        }
        cursor: str | None = None
        page = 0
        batch_rows = 0
        while True:
            page += 1
            params = dict(base_params)
            if cursor:
                params["qopts.cursor_id"] = cursor
            url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
            payload = _http_get_json(url)
            datatable = payload.get("datatable", {})
            column_names = [c["name"] for c in datatable.get("columns", [])]
            data = datatable.get("data", [])
            if data:
                frames.append(pd.DataFrame(data, columns=column_names))
            batch_rows += len(data)
            cursor = (payload.get("meta") or {}).get("next_cursor_id")
            if not cursor:
                break
            time.sleep(page_pause_seconds)
        total_rows += batch_rows
        print(
            f"  batch {batch_idx}/{len(batches)} ({len(batch)} tickers, {page} page(s)): "
            f"{batch_rows} rows  (cumulative {total_rows})"
        )
        time.sleep(page_pause_seconds)
    if not frames:
        return pd.DataFrame(columns=SF1_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for date_col in ("calendardate", "datekey", "reportperiod", "lastupdated"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    return df


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-pull SHARADAR/SF1 for the configured universe."
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        help="Sharadar ticker map JSON (output of build_sharadar_ticker_map.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for per-dimension parquet files.",
    )
    parser.add_argument(
        "--dimension",
        action="append",
        choices=["ARQ", "ARY", "ART", "MRQ", "MRY", "MRT"],
        help="SF1 dimension(s) to pull. Defaults to ARQ only. Pass multiple times for more.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Tickers per request batch.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Pull but do not write to disk.")
    args = parser.parse_args(argv)

    dimensions = args.dimension or ["ARQ"]

    _load_dotenv(args.env_file)
    api_key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not api_key:
        print(
            f"ERROR: {ENV_KEY_NAME} not set (looked in {args.env_file} and environment).",
            file=sys.stderr,
        )
        return 2

    print(f"Loading ticker map from {args.map} ...")
    map_payload = json.loads(args.map.read_text(encoding="utf-8"))
    mapping = map_payload.get("mapping", {})
    tickers = sorted(
        {row["sharadar_ticker"] for row in mapping.values() if row.get("sharadar_ticker")}
    )
    print(f"  {len(tickers)} tickers in map.")
    if not tickers:
        print("ERROR: ticker map is empty.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    overall_summary: list[dict[str, Any]] = []
    for dim in dimensions:
        print(f"\nFetching SHARADAR/SF1 dimension={dim} for {len(tickers)} tickers ...")
        t0 = time.monotonic()
        df = fetch_sf1_for_tickers(api_key, tickers, dim, batch_size=args.batch_size)
        elapsed = time.monotonic() - t0

        # Light type coercion + integrity reporting before write.
        df = _coerce_types(df)
        distinct_tickers = df["ticker"].nunique() if "ticker" in df.columns and not df.empty else 0
        date_min = df["datekey"].min() if "datekey" in df.columns and not df.empty else None
        date_max = df["datekey"].max() if "datekey" in df.columns and not df.empty else None

        out_path = args.out / f"sf1_{dim.lower()}.parquet"
        if args.dry_run:
            print(f"  --dry-run: would write {out_path} ({len(df):,} rows)")
        else:
            _write_parquet_atomic(df, out_path)
            print(f"  wrote {out_path} ({len(df):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")

        print(
            f"  rows={len(df):,}  distinct_tickers={distinct_tickers}  "
            f"datekey={date_min}..{date_max}  elapsed={elapsed:.1f}s"
        )
        overall_summary.append(
            {
                "dimension": dim,
                "rows": int(len(df)),
                "distinct_tickers": int(distinct_tickers),
                "datekey_min": str(date_min) if date_min else None,
                "datekey_max": str(date_max) if date_max else None,
                "elapsed_seconds": round(elapsed, 1),
                "output": str(out_path),
            }
        )

    if not args.dry_run:
        manifest = {
            "schema_version": 1,
            "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "source_map": str(args.map.relative_to(PROJECT_ROOT))
            if args.map.is_absolute()
            else str(args.map),
            "ticker_count": len(tickers),
            "columns": SF1_COLUMNS,
            "dimensions": overall_summary,
        }
        manifest_path = args.out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
