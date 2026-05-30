"""Paced, resumable remainder-backfill for stale daily bars.

The bulk Tiingo ingest (``-m quant_platform ingest --data-source vendor``) brings
most names current but trips Tiingo's hourly request cap (~164/run) and circuit
breaker, leaving a tail stale. This script fills ONLY the stale names, paced to
respect Polygon's free-tier 5 req/min limit (one name per ``--interval`` seconds),
storing incrementally so it is fully resumable (re-running skips now-fresh names).

Reuses the platform's vendor fetcher + ParquetBarStore (faithful mapping). Tries
Polygon by default (Tiingo is usually hour-capped right after the bulk run); pass
``--vendor tiingo`` once its hour resets to drain the rest faster.

    python scripts/backfill_stale_bars.py            # Polygon, paced
    python scripts/backfill_stale_bars.py --vendor tiingo --interval 1
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import sys
import time
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd  # noqa: E402

from quant_platform.application.operator.cli_inputs import load_instrument_contracts  # noqa: E402
from quant_platform.bootstrap.session.components import build_contract_master  # noqa: E402
from quant_platform.config import PlatformSettings  # noqa: E402
from quant_platform.services.data_service.feeds.ingest_bar_fetcher_factory import (  # noqa: E402
    build_vendor_bar_fetcher,
)
from quant_platform.services.data_service.stores.parquet_bar_store import (  # noqa: E402
    ParquetBarStore,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
BARS = PROJECT_ROOT / "data" / "parquet" / "bars"


def _max_2026_ts(instrument_id: str) -> pd.Timestamp | None:
    files = glob.glob(str(BARS / instrument_id / "daily" / "2026.parquet"))
    if not files:
        return None
    ts = pd.to_datetime(pd.read_parquet(files[0], columns=["timestamp"])["timestamp"], utc=True)
    return ts.max() if len(ts) else None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=["polygon", "tiingo"], default="polygon")
    parser.add_argument(
        "--cutoff", default="2026-05-26", help="names with max bar < this are stale"
    )
    parser.add_argument("--start", default="2026-05-12")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--interval", type=float, default=13.0, help="seconds between fetches")
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--max-passes", type=int, default=4)
    args = parser.parse_args()

    import os

    os.environ["QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN"] = f'["{args.vendor}"]'
    settings = PlatformSettings()
    fetcher = build_vendor_bar_fetcher(settings, bar_seconds=86400)
    if fetcher is None:
        print(f"ERROR: no {args.vendor} fetcher configured (token missing)", file=sys.stderr)
        return 2
    store = ParquetBarStore(settings.storage.object_store_root)
    contracts = load_instrument_contracts(UNIVERSE)
    master = build_contract_master(contracts)
    instruments = master.list_active()
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    print(f"[backfill] vendor={args.vendor} cutoff={args.cutoff} window={args.start}..{args.end}")
    t0 = time.time()
    for pass_i in range(1, args.max_passes + 1):
        stale = [
            i
            for i in instruments
            if (m := _max_2026_ts(str(i.instrument_id))) is None or m < cutoff
        ]
        print(f"[backfill] pass {pass_i}: {len(stale)} stale of {len(instruments)}")
        if not stale:
            break
        got = 0
        for idx, inst in enumerate(stale, 1):
            if (time.time() - t0) / 60.0 > args.max_minutes:
                print(f"[backfill] max-minutes ({args.max_minutes}) reached; stopping")
                break
            try:
                bars = await fetcher([inst], start, end)
            except Exception as exc:  # noqa: BLE001 - best-effort backfill
                print(f"  {inst.symbol}: ERR {type(exc).__name__} {exc}")
                bars = []
            if bars:
                await store.store_bars(bars)
                got += 1
            if idx % 10 == 0 or bars:
                print(f"  [{idx}/{len(stale)}] {inst.symbol}: {len(bars)} bars (pass total {got})")
            await asyncio.sleep(args.interval)
        print(f"[backfill] pass {pass_i} done: filled {got}")
        if got == 0:
            print("[backfill] no progress this pass; stopping (vendor likely capped)")
            break

    remaining = sum(
        1 for i in instruments if (m := _max_2026_ts(str(i.instrument_id))) is None or m < cutoff
    )
    fresh = len(instruments) - remaining
    print(
        f"\n[backfill] DONE: {fresh}/{len(instruments)} fresh (>= {args.cutoff}); {remaining} stale"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
