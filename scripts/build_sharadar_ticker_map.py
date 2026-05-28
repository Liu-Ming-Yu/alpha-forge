"""Build a Sharadar (Nasdaq Data Link) ticker map for the configured universe.

Pulls ``SHARADAR/TICKERS`` (table=SF1 rows — the Core US Fundamentals universe),
joins it against the symbols in a contracts file (e.g.
``infra/config/universe_300.json``), and writes the resulting per-instrument map
to a JSON file.

The output is keyed by the same ``instrument_id`` UUIDs the rest of the platform
uses, so downstream feature code can join (instrument_id, datekey) without ever
touching raw symbols.

Drops are reported in two buckets:

* **no_sf1_row** — the symbol exists nowhere in ``SHARADAR/TICKERS`` (table=SF1)
  for the active dataset. Either the ticker has been changed/retired or it's
  outside Sharadar coverage.
* **stale** — the SF1 row exists but ``lastupdated`` is older than the configured
  freshness window (default 180 days). Indicates a name that has effectively
  stopped reporting on the Sharadar feed.

Usage::

    python scripts/build_sharadar_ticker_map.py \\
        --universe infra/config/universe_300.json \\
        --out infra/config/sharadar_ticker_map.json

The Nasdaq Data Link key is read from the environment variable
``QP__DATA_INGEST__NASDAQ_DATA_LINK_API_KEY``. If a ``.env`` file is present in
the project root it is read inline (no python-dotenv dependency).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
DEFAULT_OUT = PROJECT_ROOT / "infra" / "config" / "sharadar_ticker_map.json"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/TICKERS"
ENV_KEY_NAME = "QP__DATA_INGEST__NASDAQ_DATA_LINK_API_KEY"

# Columns we want from SHARADAR/TICKERS. Keeping this list small reduces payload
# size and makes the parser tolerant of upstream schema additions.
TICKER_COLUMNS = [
    "table",
    "permaticker",
    "ticker",
    "name",
    "exchange",
    "isdelisted",
    "category",
    "sector",
    "industry",
    "scalemarketcap",
    "currency",
    "firstpricedate",
    "lastpricedate",
    "firstquarter",
    "lastquarter",
    "lastupdated",
]


def _load_dotenv(path: Path) -> None:
    """Tiny ``.env`` reader: KEY=value lines, ignores comments and blanks.

    Does not overwrite variables that are already set in the process
    environment, so a real shell ``export`` always wins over the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Issue a GET, return decoded JSON. Raises on non-200."""
    # URL is built from API_BASE (a pinned https host) plus urlencoded params,
    # never user-controlled scheme — S310's file:/custom-scheme worry is moot.
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "quant-platform/build-sharadar-map"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}")
        body = resp.read()
    return json.loads(body)


def fetch_sf1_tickers(api_key: str, *, page_pause_seconds: float = 0.3) -> list[dict[str, Any]]:
    """Pull every ``SHARADAR/TICKERS`` row where ``table=SF1``.

    Sharadar's datatables API paginates via ``next_cursor_id``. The SF1 ticker
    universe is small (~14k rows as of writing), so a single pass with a short
    courtesy sleep between pages stays well inside any rate limit.
    """
    columns = ",".join(TICKER_COLUMNS)
    base_params = {
        "table": "SF1",
        "qopts.columns": columns,
        "api_key": api_key,
    }

    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0
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
        for raw_row in data:
            rows.append(dict(zip(column_names, raw_row, strict=True)))

        cursor = (payload.get("meta") or {}).get("next_cursor_id")
        print(
            f"  page {page}: {len(data)} rows (cumulative {len(rows)}); "
            f"cursor={'yes' if cursor else 'no'}"
        )
        if not cursor:
            break
        time.sleep(page_pause_seconds)
    return rows


def load_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Universe file {path} is not a JSON object")
    return payload


def build_map(
    universe: dict[str, dict[str, Any]],
    sf1_rows: list[dict[str, Any]],
    *,
    staleness_days: int,
    as_of: _dt.date,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Join universe symbols against SF1 rows; return (map, drops)."""
    by_ticker: dict[str, dict[str, Any]] = {}
    for row in sf1_rows:
        ticker = (row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        # If a ticker recycles across companies, the row whose `lastquarter`
        # extends nearest to today is the live mapping. Compare by lastupdated.
        existing = by_ticker.get(ticker)
        if existing is None:
            by_ticker[ticker] = row
            continue
        existing_lu = existing.get("lastupdated") or ""
        new_lu = row.get("lastupdated") or ""
        if new_lu > existing_lu:
            by_ticker[ticker] = row

    mapped: dict[str, dict[str, Any]] = {}
    drops: dict[str, list[str]] = {"no_sf1_row": [], "stale": [], "delisted": []}

    for instrument_id, contract in universe.items():
        symbol = (contract.get("symbol") or "").strip().upper()
        if not symbol:
            drops.setdefault("no_symbol", []).append(instrument_id)
            continue
        row = by_ticker.get(symbol)
        if row is None:
            drops["no_sf1_row"].append(symbol)
            continue

        last_updated = _parse_iso_date(row.get("lastupdated"))
        is_stale = last_updated is not None and (as_of - last_updated).days > staleness_days
        is_delisted = bool(
            row.get("isdelisted") and str(row.get("isdelisted")).upper().startswith("Y")
        )
        if is_delisted:
            drops["delisted"].append(symbol)
            continue
        if is_stale:
            drops["stale"].append(symbol)
            continue

        mapped[instrument_id] = {
            "symbol": symbol,
            "sharadar_ticker": row.get("ticker"),
            "permaticker": row.get("permaticker"),
            "name": row.get("name"),
            "exchange": row.get("exchange"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "scale_marketcap": row.get("scalemarketcap"),
            "currency": row.get("currency"),
            "first_quarter": row.get("firstquarter"),
            "last_quarter": row.get("lastquarter"),
            "first_price_date": row.get("firstpricedate"),
            "last_price_date": row.get("lastpricedate"),
            "lastupdated": row.get("lastupdated"),
        }
    return mapped, drops


def _parse_iso_date(value: object) -> _dt.date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return _dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Sharadar SF1 ticker map for the configured universe."
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--staleness-days",
        type=int,
        default=180,
        help="Drop names whose SF1 row hasn't updated in more than this many days.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Reference date for staleness check (ISO YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print summary, do not write the output file."
    )
    args = parser.parse_args(argv)

    _load_dotenv(args.env_file)
    api_key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not api_key:
        print(
            f"ERROR: {ENV_KEY_NAME} not set (looked in {args.env_file} and environment).",
            file=sys.stderr,
        )
        return 2

    as_of = _dt.date.fromisoformat(args.as_of) if args.as_of else _dt.date.today()

    print(f"Loading universe from {args.universe} ...")
    universe = load_universe(args.universe)
    print(f"  {len(universe)} instruments in universe.")

    print("Fetching SHARADAR/TICKERS (table=SF1) ...")
    sf1_rows = fetch_sf1_tickers(api_key)
    print(f"  {len(sf1_rows)} SF1 rows fetched.")

    mapped, drops = build_map(universe, sf1_rows, staleness_days=args.staleness_days, as_of=as_of)

    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "universe_path": str(args.universe.relative_to(PROJECT_ROOT))
        if args.universe.is_absolute()
        else str(args.universe),
        "as_of": as_of.isoformat(),
        "staleness_days": args.staleness_days,
        "counts": {
            "universe": len(universe),
            "mapped": len(mapped),
            **{f"dropped_{k}": len(v) for k, v in drops.items()},
        },
        "drops": drops,
        "mapping": mapped,
    }

    print("Summary:")
    for k, v in payload["counts"].items():
        print(f"  {k}: {v}")
    if any(drops.values()):
        for bucket, names in drops.items():
            if names:
                preview = ", ".join(sorted(names)[:20])
                more = "" if len(names) <= 20 else f" (+{len(names) - 20} more)"
                print(f"  drop[{bucket}]: {preview}{more}")

    if args.dry_run:
        print("--dry-run: not writing output.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(args.out, payload)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
