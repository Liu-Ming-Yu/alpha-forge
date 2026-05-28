"""Build a symbol -> SEC CIK map for a universe contracts file.

Pulls SEC's authoritative ``company_tickers.json`` (free, public) and emits a
``{symbol: cik}`` JSON consumable by ``text-events ingest-sec --cik-map-file``.

Usage::

    python scripts/build_cik_map.py \
        --universe infra/config/universe_300.json \
        --out infra/config/sec_cik_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Verified CIKs for symbols SEC's company_tickers.json omits — acquired/
# delisted names (DFS, HES) keep historical filings, and a few active names
# carry no ticker in that file. Each CIK was confirmed against
# data.sec.gov/submissions/CIK{cik}.json.
_MANUAL_CIK: dict[str, str] = {
    "CTRA": "858470",  # Coterra Energy Inc.
    "DFS": "1393612",  # Discover Financial Services
    "HES": "4447",  # Hess Corp
    "HOLX": "859737",  # Hologic Inc
    "IPG": "51644",  # Interpublic Group
    "MMC": "62709",  # Marsh & McLennan Companies
}


def _user_agent(explicit: str | None) -> str:
    """Resolve the SEC User-Agent (SEC requires a descriptive contact string)."""
    if explicit:
        return explicit
    env_key = "QP__DATA_INGEST__SEC_USER_AGENT"
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{env_key}="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return "QuantPlatform research contact@example.com"


def fetch_sec_ticker_ciks(user_agent: str) -> dict[str, str]:
    """Return an upper-case ticker -> CIK (digit string) map from SEC."""
    resp = httpx.get(_SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=60.0)
    if resp.status_code != 200:
        raise SystemExit(f"SEC company_tickers request failed: HTTP {resp.status_code}")
    payload = resp.json()
    out: dict[str, str] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = str(int(cik))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a symbol -> SEC CIK map.")
    parser.add_argument("--universe", default="infra/config/universe_300.json")
    parser.add_argument("--out", default="infra/config/sec_cik_map.json")
    parser.add_argument("--user-agent", default=None)
    args = parser.parse_args()

    universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    symbols = sorted({str(spec["symbol"]).upper() for spec in universe.values()})

    ticker_ciks = fetch_sec_ticker_ciks(_user_agent(args.user_agent))

    cik_map: dict[str, str] = {}
    missing: list[str] = []
    for symbol in symbols:
        # SEC uses '-' for class shares (e.g. BRK-B); try both forms, then
        # fall back to the verified manual overrides.
        cik = (
            ticker_ciks.get(symbol)
            or ticker_ciks.get(symbol.replace(".", "-"))
            or _MANUAL_CIK.get(symbol)
        )
        if cik:
            cik_map[symbol] = cik
        else:
            missing.append(symbol)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(dict(sorted(cik_map.items())), indent=2), encoding="utf-8")
    print(f"Wrote {len(cik_map)}/{len(symbols)} symbol->CIK mappings to {out_path}")
    if missing:
        print(f"Unmatched ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
