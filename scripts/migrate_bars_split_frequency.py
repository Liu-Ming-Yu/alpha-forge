"""Migrate legacy mixed-frequency bar parquet files into the daily/intraday layout.

Walks ``data/parquet/bars/<instrument_id>/`` for any top-level ``<year>.parquet``
files (the pre-split layout) and splits each one into the post-split layout::

    bars/<inst>/<year>.parquet           # legacy, removed after migration
      ->
    bars/<inst>/daily/<year>.parquet     # bar_seconds == 86400 rows
    bars/<inst>/intraday/<year>.parquet  # bar_seconds != 86400 rows

The migration is idempotent (a second run finds no legacy files and exits
clean) and safe to run while writers are active: it reuses
``ParquetBarStore._migrate_legacy_partition_if_present``, which takes the
same exclusive file lock that the write path uses. Empty buckets are not
created -- a daily-only legacy file produces only the ``daily/`` file.

The store's write path will perform this same migration lazily on the next
write to each ``(instrument, year)`` partition; this script is a way to do
the whole back-catalog in one pass instead of waiting for organic writes.

Usage::

    # Default: dry-run report against ./data/parquet
    python scripts/migrate_bars_split_frequency.py

    # Apply migration
    python scripts/migrate_bars_split_frequency.py --apply

    # Point at a non-default object-store root
    python scripts/migrate_bars_split_frequency.py --root /mnt/quant/parquet --apply

    # Limit to a single instrument
    python scripts/migrate_bars_split_frequency.py --instrument <uuid> --apply
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_platform.services.data_service.stores.parquet_bar_store import (  # noqa: E402
    DAILY_BAR_SECONDS,
    ParquetBarStore,
)
from quant_platform.services.data_service.stores.parquet_store_io import (  # noqa: E402
    BAR_SCHEMA,
)


class LegacyFile(NamedTuple):
    instrument_id: uuid.UUID
    year: int
    path: Path
    total_rows: int
    daily_rows: int
    intraday_rows: int


def _find_legacy_files(
    root: Path,
    only_instrument: uuid.UUID | None,
) -> list[LegacyFile]:
    """Discover legacy ``bars/<inst>/<year>.parquet`` files under ``root``."""
    bars_root = root / "bars"
    if not bars_root.exists():
        return []

    out: list[LegacyFile] = []
    for inst_dir in sorted(bars_root.iterdir()):
        if not inst_dir.is_dir():
            continue
        try:
            inst_id = uuid.UUID(inst_dir.name)
        except ValueError:
            continue
        if only_instrument is not None and inst_id != only_instrument:
            continue
        for child in sorted(inst_dir.iterdir()):
            if not child.is_file() or child.suffix != ".parquet":
                continue
            try:
                year = int(child.stem)
            except ValueError:
                continue
            table = pq.read_table(child, schema=BAR_SCHEMA, columns=["bar_seconds"])
            bs = table.column("bar_seconds").to_pylist()
            daily = sum(1 for v in bs if v == DAILY_BAR_SECONDS)
            intraday = len(bs) - daily
            out.append(
                LegacyFile(
                    instrument_id=inst_id,
                    year=year,
                    path=child,
                    total_rows=len(bs),
                    daily_rows=daily,
                    intraday_rows=intraday,
                )
            )
    return out


def _print_report(legacy: list[LegacyFile]) -> None:
    if not legacy:
        print("no legacy files found")
        return
    total_files = len(legacy)
    total_rows = sum(f.total_rows for f in legacy)
    total_daily = sum(f.daily_rows for f in legacy)
    total_intraday = sum(f.intraday_rows for f in legacy)
    insts = {f.instrument_id for f in legacy}
    print(f"found {total_files} legacy files across {len(insts)} instruments")
    print(f"  total rows: {total_rows:,}")
    print(f"  daily rows: {total_daily:,}")
    print(f"  intraday rows: {total_intraday:,}")
    mixed = [f for f in legacy if f.daily_rows > 0 and f.intraday_rows > 0]
    if mixed:
        print(f"  mixed (both buckets in one file): {len(mixed)}")
        for f in mixed[:5]:
            print(
                f"    {f.instrument_id} {f.year}.parquet "
                f"daily={f.daily_rows} intraday={f.intraday_rows}"
            )
        if len(mixed) > 5:
            print(f"    ... and {len(mixed) - 5} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "parquet",
        help="Object-store root containing the bars/ subdirectory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--instrument",
        type=uuid.UUID,
        default=None,
        help="Restrict migration to a single instrument UUID.",
    )
    args = parser.parse_args()

    legacy = _find_legacy_files(args.root, args.instrument)
    _print_report(legacy)

    if not args.apply:
        if legacy:
            print("\ndry-run only -- pass --apply to perform the migration")
        return 0

    if not legacy:
        return 0

    store = ParquetBarStore(args.root)
    migrated = 0
    for entry in legacy:
        store._migrate_legacy_partition_if_present(entry.instrument_id, entry.year)  # noqa: SLF001
        migrated += 1
        if migrated % 25 == 0:
            print(f"  migrated {migrated}/{len(legacy)}")
    print(f"migrated {migrated} legacy files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
