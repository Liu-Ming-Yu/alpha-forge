"""Scan structured logs for fail-closed / failure events and emit an operator digest.

The platform standard is snake_case dotted event names (e.g. ``text_extractor.failure``,
``cycle_guards.fail_closed``, ``broker.disconnected``). This script tails any JSON-line
log files and surfaces *unresolved* incidents to a human-readable channel — exit
code is non-zero when fresh failures are detected so cron / Task Scheduler can
trigger an actual alert.

Usage::

    python scripts/alert_failures.py logs/*.log                # scan + digest
    python scripts/alert_failures.py --since-minutes 60 logs/  # last hour only
    python scripts/alert_failures.py --json logs/              # machine-readable

Designed to be wrapped by cron / Windows Task Scheduler::

    */5 * * * * python scripts/alert_failures.py --since-minutes 5 logs/ \
        || curl -X POST -d @- https://your-webhook/alerts

Conventions honored:
  * Reads JSON-line structlog output (one event per line, dotted snake_case ``event``).
  * Treats any event ending in ``.failure``, ``.fail_closed``, ``.halted``, ``.error``
    or carrying ``log.level=error`` as an alertable incident.
  * Severity inferred from suffix: ``fail_closed``/``halted`` → CRITICAL,
    ``failure``/``error`` → ERROR. ``warning`` level is reported but does not
    flip exit code.

Exit codes:
  0 — no fresh failures in the window
  1 — fresh failures detected (cron should fire the actual alert)
  2 — bad arguments / no log files matched
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CRITICAL_SUFFIXES = (".fail_closed", ".halted", ".kill_switch")
ERROR_SUFFIXES = (".failure", ".error", ".disconnected", ".rejected")
SEVERITY_ORDER = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}


def _classify(event: str, level: str) -> str | None:
    if any(event.endswith(suffix) for suffix in CRITICAL_SUFFIXES):
        return "CRITICAL"
    if any(event.endswith(suffix) for suffix in ERROR_SUFFIXES):
        return "ERROR"
    if level.lower() in {"critical", "fatal"}:
        return "CRITICAL"
    if level.lower() == "error":
        return "ERROR"
    if level.lower() == "warning":
        return "WARNING"
    return None


def _parse_ts(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("timestamp") or payload.get("ts") or payload.get("time")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _expand(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.log")))
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif "*" in p or "?" in p:
            expanded.extend(Path(match) for match in glob.glob(p, recursive=True))
        elif path.exists():
            expanded.append(path)
    return expanded


def _scan(paths: list[Path], since: datetime | None) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event = str(payload.get("event") or "")
                    level = str(payload.get("level") or payload.get("log_level") or "")
                    severity = _classify(event, level)
                    if severity is None:
                        continue
                    ts = _parse_ts(payload)
                    if since is not None and (ts is None or ts < since):
                        continue
                    incidents.append(
                        {
                            "severity": severity,
                            "event": event or "(unknown)",
                            "timestamp": ts.isoformat() if ts else None,
                            "source_file": str(path),
                            "message": payload.get("message") or payload.get("msg") or "",
                            "context": {
                                k: v
                                for k, v in payload.items()
                                if k not in {"event", "level", "timestamp", "ts", "time"}
                            },
                        }
                    )
        except OSError as exc:
            incidents.append(
                {
                    "severity": "ERROR",
                    "event": "alert_failures.scan_error",
                    "timestamp": None,
                    "source_file": str(path),
                    "message": str(exc),
                    "context": {},
                }
            )
    return incidents


def _render_digest(incidents: list[dict[str, Any]]) -> str:
    if not incidents:
        return "[alert_failures] no fresh failures in window."
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inc in incidents:
        by_event[inc["event"]].append(inc)
    counts = Counter(inc["severity"] for inc in incidents)
    lines: list[str] = []
    lines.append(
        f"[alert_failures] {len(incidents)} incidents — "
        f"CRITICAL={counts.get('CRITICAL', 0)} "
        f"ERROR={counts.get('ERROR', 0)} "
        f"WARNING={counts.get('WARNING', 0)}"
    )

    def _sort_key(name: str) -> tuple[int, str]:
        return (SEVERITY_ORDER.get(by_event[name][0]["severity"], 9), name)

    for event in sorted(by_event, key=_sort_key):
        bucket = by_event[event]
        first = bucket[0]
        last_ts = max((b.get("timestamp") or "" for b in bucket), default="")
        lines.append(f"  [{first['severity']}] {event}  (×{len(bucket)}, last={last_ts or 'n/a'})")
        if first.get("message"):
            lines.append(f"      msg: {first['message']}")
        for key in ("instrument_id", "model_version", "cycle_id", "exc_type"):
            value = first["context"].get(key)
            if value is not None:
                lines.append(f"      {key}: {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Log files, glob patterns, or directories.")
    parser.add_argument(
        "--since-minutes",
        type=int,
        default=0,
        help="Only count incidents with timestamps within the last N minutes (0 = no filter).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    files = _expand(args.paths)
    if not files:
        print(f"[alert_failures] no log files matched: {args.paths!r}", file=sys.stderr)
        return 2

    since: datetime | None = None
    if args.since_minutes > 0:
        since = datetime.now(UTC) - timedelta(minutes=args.since_minutes)

    incidents = _scan(files, since)
    fresh_failures = [i for i in incidents if i["severity"] in {"CRITICAL", "ERROR"}]

    if args.json:
        print(json.dumps({"incidents": incidents, "fresh_failures": len(fresh_failures)}, indent=2))
    else:
        print(_render_digest(incidents))

    return 1 if fresh_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
