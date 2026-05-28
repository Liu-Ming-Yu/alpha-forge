"""Build a liquid US-equity universe contracts file from Polygon.io reference data.

Selects the most liquid US common stocks by trailing average dollar volume and
emits a contracts JSON in the same schema as ``infra/config/paper_contracts.json``.

Usage::

    python scripts/build_universe.py --out infra/config/universe_300.json --count 300

The Polygon API key is read from ``QP__DATA_INGEST__POLYGON_API_KEY``
(environment or ``.env``), or passed with ``--api-key``.

``con_id`` is emitted as ``0``: Polygon has no Interactive Brokers contract ids.
Resolve them with a one-time IB ``reqContractDetails`` pass before using the
universe for live/paper order routing.  Historical ingest, feature backfill and
backtests key off ``symbol`` / ``instrument_id`` and do not need ``con_id``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import httpx

# Stable namespace for instrument_id UUIDs across all contracts files (the
# 15-name paper smoke fixture and the expanded universes).  Do not change.
NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
BASE = "https://api.polygon.io"

# Polygon primary_exchange MIC -> human exchange name.  Only the major US
# listing venues are kept; everything else is dropped.
_EXCHANGE_BY_MIC = {
    "XNYS": "NYSE",
    "XNAS": "NASDAQ",
    "XASE": "AMEX",
    "ARCX": "NYSE",
    "BATS": "NYSE",
}

# SIC code ranges -> GICS-style sector, most-specific ranges first.  Coarse but
# sufficient for sector-cap enforcement (QP__RISK__REQUIRE_SECTOR_MAPPING).
_SIC_SECTORS: list[tuple[int, int, str]] = [
    (2833, 2836, "Health Care"),  # pharmaceutical preparations
    (3826, 3851, "Health Care"),  # medical / lab instruments
    (8000, 8099, "Health Care"),  # health services
    (2835, 2835, "Health Care"),  # in-vitro diagnostics
    (3570, 3579, "Information Technology"),
    (3670, 3679, "Information Technology"),
    (3661, 3669, "Information Technology"),
    (7370, 7379, "Information Technology"),
    (2700, 2799, "Communication Services"),
    (4800, 4899, "Communication Services"),
    (7800, 7841, "Communication Services"),
    (4900, 4999, "Utilities"),
    (6000, 6299, "Financials"),
    (6300, 6411, "Financials"),
    (6700, 6799, "Financials"),
    (6500, 6599, "Real Estate"),
    (1200, 1399, "Energy"),
    (2900, 2999, "Energy"),
    (1000, 1099, "Materials"),
    (1400, 1499, "Materials"),
    (2400, 2499, "Materials"),
    (2600, 2699, "Materials"),
    (2800, 2829, "Materials"),
    (3200, 3399, "Materials"),
    (2000, 2199, "Consumer Staples"),
    (2840, 2844, "Consumer Staples"),
    (5400, 5499, "Consumer Staples"),
    (5912, 5912, "Consumer Staples"),
    (2300, 2399, "Consumer Discretionary"),
    (2500, 2599, "Consumer Discretionary"),
    (3000, 3199, "Consumer Discretionary"),
    (3630, 3669, "Consumer Discretionary"),
    (3710, 3716, "Consumer Discretionary"),
    (5000, 5999, "Consumer Discretionary"),
    (7000, 7299, "Consumer Discretionary"),
    (7900, 7999, "Consumer Discretionary"),
    (1500, 1799, "Industrials"),
    (3400, 3569, "Industrials"),
    (3580, 3629, "Industrials"),
    (3700, 3799, "Industrials"),
    (4000, 4799, "Industrials"),
]


def _load_api_key(explicit: str | None) -> str:
    """Resolve the Polygon API key from --api-key, env, or .env."""
    import os

    if explicit:
        return explicit.strip()
    env_key = "QP__DATA_INGEST__POLYGON_API_KEY"
    if os.environ.get(env_key):
        return os.environ[env_key].strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{env_key}="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(f"Polygon API key not found. Set {env_key} in .env or pass --api-key.")


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> dict:
    """GET with light retry on transient rate-limit / server errors."""
    for attempt, delay in enumerate((0.0, 2.0, 5.0, 15.0)):
        if delay:
            time.sleep(delay)
        resp = client.get(url, params=params, timeout=60.0)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 502, 503, 504):
            print(f"  retry {attempt + 1} (HTTP {resp.status_code})", file=sys.stderr)
            continue
        raise SystemExit(f"Polygon request failed: HTTP {resp.status_code} {resp.url}")
    raise SystemExit(f"Polygon request failed after retries: {url}")


def _sector_for_sic(sic_code: object) -> str:
    """Map a Polygon SIC code to a GICS-style sector; default Industrials."""
    try:
        code = int(str(sic_code))
    except (TypeError, ValueError):
        return "Industrials"
    for lo, hi, sector in _SIC_SECTORS:
        if lo <= code <= hi:
            return sector
    return "Industrials"


def _fetch_liquidity(client: httpx.Client, key: str, days: int) -> dict[str, dict]:
    """Average volume and last close per ticker over recent grouped daily bars."""
    agg: dict[str, dict] = {}
    populated = 0
    cursor = date.today() - timedelta(days=1)
    scanned = 0
    while populated < days and scanned < days * 3 + 10:
        scanned += 1
        url = f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{cursor.isoformat()}"
        body = _get(client, url, {"adjusted": "true", "apiKey": key})
        results = body.get("results") or []
        cursor -= timedelta(days=1)
        if not results:
            continue  # weekend / holiday
        populated += 1
        for row in results:
            ticker = row.get("T")
            close = row.get("c")
            vol = row.get("v")
            if not ticker or close is None or vol is None:
                continue
            entry = agg.setdefault(ticker, {"vol": 0.0, "n": 0, "last_close": None})
            entry["vol"] += float(vol)
            entry["n"] += 1
            if entry["last_close"] is None:  # first (most recent) day seen
                entry["last_close"] = float(close)
    print(f"  scanned {populated} trading days, {len(agg)} tickers", file=sys.stderr)
    return agg


def _fetch_common_stocks(client: httpx.Client, key: str) -> dict[str, dict]:
    """All active US common stocks keyed by ticker (ticker -> reference fields)."""
    out: dict[str, dict] = {}
    url = f"{BASE}/v3/reference/tickers"
    params = {
        "type": "CS",
        "market": "stocks",
        "active": "true",
        "limit": "1000",
        "apiKey": key,
    }
    while True:
        body = _get(client, url, params)
        for row in body.get("results") or []:
            ticker = row.get("ticker")
            if ticker:
                out[ticker] = row
        next_url = body.get("next_url")
        if not next_url:
            break
        url, params = next_url, {"apiKey": key}
    print(f"  {len(out)} active common stocks in reference", file=sys.stderr)
    return out


def _fetch_sector(client: httpx.Client, key: str, ticker: str) -> str:
    """Per-ticker SIC lookup -> GICS sector."""
    body = _get(client, f"{BASE}/v3/reference/tickers/{ticker}", {"apiKey": key})
    return _sector_for_sic((body.get("results") or {}).get("sic_code"))


def build_universe(api_key: str, count: int, min_price: float, lookback_days: int) -> dict:
    """Select the top liquid common stocks and emit a contracts mapping."""
    with httpx.Client() as client:
        print("Fetching recent liquidity (grouped daily bars)...", file=sys.stderr)
        liquidity = _fetch_liquidity(client, api_key, lookback_days)
        print("Fetching common-stock reference list...", file=sys.stderr)
        reference = _fetch_common_stocks(client, api_key)

        ranked: list[tuple[float, str, float, float]] = []
        for ticker, ref in reference.items():
            mic = ref.get("primary_exchange")
            if mic not in _EXCHANGE_BY_MIC:
                continue
            liq = liquidity.get(ticker)
            if not liq or not liq["n"] or liq["last_close"] is None:
                continue
            close = liq["last_close"]
            if close < min_price:
                continue
            avg_vol = liq["vol"] / liq["n"]
            dollar_vol = avg_vol * close
            ranked.append((dollar_vol, ticker, avg_vol, close))

        ranked.sort(key=lambda r: r[0], reverse=True)
        selected = ranked[:count]
        print(f"Selected {len(selected)} names; resolving sectors...", file=sys.stderr)

        contracts: dict[str, dict] = {}
        for idx, (_dollar_vol, ticker, avg_vol, close) in enumerate(selected, start=1):
            sector = _fetch_sector(client, api_key, ticker)
            ref = reference[ticker]
            uid = str(uuid.uuid5(NS, ticker))
            contracts[uid] = {
                "symbol": ticker,
                "exchange": "SMART",
                "primary_exchange": _EXCHANGE_BY_MIC[ref["primary_exchange"]],
                "currency": "USD",
                "sec_type": "STK",
                "asset_class": "EQUITY",
                "sector": sector,
                "active": True,
                "lot_size": 1,
                "con_id": 0,
                "adv_shares_20d": int(avg_vol),
                "last_close": round(close, 2),
            }
            if idx % 50 == 0:
                print(f"  resolved {idx}/{len(selected)}", file=sys.stderr)
    return contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a liquid US-equity universe.")
    parser.add_argument("--out", default="infra/config/universe_300.json")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = _load_api_key(args.api_key)
    contracts = build_universe(api_key, args.count, args.min_price, args.lookback_days)
    if not contracts:
        raise SystemExit("No instruments selected — check API key and filters.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contracts, indent=2), encoding="utf-8")
    print(f"Wrote {len(contracts)} contracts to {out_path}")
    sectors: dict[str, int] = {}
    for spec in contracts.values():
        sectors[spec["sector"]] = sectors.get(spec["sector"], 0) + 1
    for sector, n in sorted(sectors.items(), key=lambda kv: -kv[1]):
        print(f"  {sector}: {n}")
    print(
        "NOTE: con_id is 0 — run a one-time IB reqContractDetails pass before live/paper routing."
    )


if __name__ == "__main__":
    main()
