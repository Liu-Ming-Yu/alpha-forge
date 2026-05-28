"""Pretty-print the LLM extraction progress JSON heartbeat.

The ``text-events extract-features --status-file <path>`` job writes the file
every ~10 seconds. This reader formats it with elapsed/ETA/rate so the operator
can monitor a long-running backfill.

Usage::

    python scripts/extract_status.py [--file PATH] [--watch]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_FILE = "data/parquet/research/text_events/extract_status.json"


def _human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(max(0, seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {s}s"
    return f"{s}s"


def _bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render(status: dict[str, object]) -> str:
    total = int(status.get("total_events", 0) or 0)
    extracted = int(status.get("extracted", 0) or 0)
    skipped_dup = int(status.get("skipped_duplicates", 0) or 0)
    skipped_role = int(status.get("skipped_document_role", 0) or 0)
    skipped_macro = int(status.get("skipped_macro", 0) or 0)
    failed = int(status.get("failed", 0) or 0)
    in_flight = int(status.get("in_flight", 0) or 0)
    elapsed = float(status.get("elapsed_seconds", 0.0) or 0.0)
    rate = float(status.get("rate_per_minute", 0.0) or 0.0)
    eta_s = status.get("eta_seconds")
    eta_seconds = float(eta_s) if isinstance(eta_s, int | float) else None
    terminal = bool(status.get("terminal"))
    updated_at = str(status.get("updated_at") or "")
    started_at = str(status.get("started_at") or "")
    processed = extracted + skipped_dup + skipped_role + skipped_macro + failed
    progress_pct = (processed / total * 100.0) if total else 0.0
    age_s: float | None = None
    if updated_at:
        try:
            ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            age_s = (datetime.now(tz=UTC) - ts).total_seconds()
        except ValueError:
            age_s = None
    stuck = age_s is not None and age_s > 60 and not terminal
    state = "DONE" if terminal else ("STUCK?" if stuck else "RUNNING")
    return "\n".join(
        [
            f"  state         {state}",
            f"  progress      {_bar(processed, total)} {processed}/{total} ({progress_pct:.1f}%)",
            f"  extracted     {extracted}",
            f"  skipped       dup={skipped_dup} role={skipped_role} macro={skipped_macro}",
            f"  failed        {failed}",
            f"  in_flight     {in_flight}",
            f"  rate          {rate:.1f}/min",
            f"  elapsed       {_human_seconds(elapsed)}",
            f"  ETA           {_human_seconds(eta_seconds)}",
            f"  started       {started_at}",
            f"  updated       {updated_at}"
            + (f"  ({_human_seconds(age_s)} ago)" if age_s is not None else ""),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read text-extraction status JSON.")
    parser.add_argument("--file", default=_DEFAULT_FILE, help="Status JSON path")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh every 10 seconds until extraction is terminal",
    )
    args = parser.parse_args()
    path = Path(args.file)

    while True:
        if not path.exists():
            print(f"No status file at {path}", file=sys.stderr)
            if not args.watch:
                sys.exit(2)
        else:
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Failed to read {path}: {exc}", file=sys.stderr)
                if not args.watch:
                    sys.exit(2)
                status = None
            if status is not None:
                print(_render(status))
                if not args.watch or status.get("terminal"):
                    return
        time.sleep(10)


if __name__ == "__main__":
    main()
