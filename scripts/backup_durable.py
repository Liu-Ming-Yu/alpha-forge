"""Daily snapshot of the durable artifacts that are expensive to regenerate.

Targets:
  * LLM replay cache (``data/parquet/research/text_events/extractions``) —
    hundreds of paid DeepSeek calls; permanent loss is real money.
  * Postgres ``text_events`` table — extraction provenance.
  * Postgres ``feature_vectors`` rows with text-family feature ids — promoted
    text-alpha state.

Outputs ``backups/durable_<UTC-timestamp>/`` containing:
  * ``replay_cache.tar.gz``         (skip if missing)
  * ``pg_text_events.sql.gz``       (skip if pg_dump unavailable / table missing)
  * ``pg_feature_vectors_text.csv.gz``
  * ``manifest.json``               (always)

Design choices vs the existing ``local_backup_drill.sh``:
  * Pure-Python so it runs on Windows dev boxes without bash/wsl.
  * Scoped to the *expensive-to-recompute* slice rather than full DB+object-store.
    Run the shell drill weekly for the heavy snapshot; run this script daily.
  * Atomic: writes to ``<dest>.partial`` then renames on success so a crashed
    cron does not leave a half-built directory that looks valid.
  * Quiet retention: keeps the most-recent ``--retain`` snapshots (default 7),
    independent of mtime, so a clock skew won't wipe everything.

Usage::

    python scripts/backup_durable.py              # writes ./backups/durable_<ts>/
    python scripts/backup_durable.py --dry-run    # show what would happen
    python scripts/backup_durable.py --retain 14  # keep 14 most-recent

Honors the standard config envvars:
  * ``QP__STORAGE__OBJECT_STORE_ROOT``    (defaults to ./data/parquet)
  * ``QP__STORAGE__POSTGRES_DSN``         (optional; skips pg dumps if unset)
  * ``QP_BACKUP_ROOT``                    (defaults to ./backups)

Exit codes:
  0 — backup written (or dry-run printed)
  2 — fatal misconfiguration (object-store path missing AND no DSN)
  3 — partial: replay cache copied but pg_dump failed
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_ROOT = REPO_ROOT / "data" / "parquet"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "backups"
REPLAY_CACHE_SUBPATH = Path("research/text_events/extractions")
BACKUP_PREFIX = "durable_"


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_env_file() -> None:
    """Lightweight .env loader so cron jobs that don't source it still work.

    Skipped if the variable is already set in the parent environment.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _pg_dsn_for_pg_dump() -> str | None:
    """Return a libpq-style DSN, stripping the SQLAlchemy driver prefix."""
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN")
    if not dsn:
        return None
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


def _archive_replay_cache(source: Path, dest: Path) -> dict[str, object]:
    if not source.exists():
        return {"status": "skipped", "reason": "source_missing", "path": str(source)}
    files = sum(1 for _ in source.rglob("*") if _.is_file())
    with tarfile.open(dest, mode="w:gz") as tar:
        tar.add(source, arcname=source.name)
    return {
        "status": "ok",
        "source": str(source),
        "archive": str(dest),
        "file_count": files,
        "size_bytes": dest.stat().st_size,
    }


def _pg_dump_table(dsn: str, table: str, dest: Path) -> dict[str, object]:
    if shutil.which("pg_dump") is None:
        return {"status": "skipped", "reason": "pg_dump_missing", "table": table}
    cmd = ["pg_dump", "--format=plain", "--data-only", "--table", table, dsn]
    try:
        with gzip.open(dest, "wb") as gz:
            proc = subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
            gz.write(proc.stdout)
    except subprocess.CalledProcessError as exc:
        if dest.exists():
            dest.unlink()
        return {
            "status": "error",
            "table": table,
            "stderr": exc.stderr.decode("utf-8", errors="ignore")[-500:],
        }
    return {
        "status": "ok",
        "table": table,
        "archive": str(dest),
        "size_bytes": dest.stat().st_size,
    }


def _pg_dump_text_feature_vectors(dsn: str, dest: Path) -> dict[str, object]:
    """Export feature_vectors rows whose feature_id starts with v10_*_text_* via COPY."""
    if shutil.which("psql") is None:
        return {"status": "skipped", "reason": "psql_missing"}
    sql = (
        "COPY (SELECT * FROM feature_vectors WHERE feature_id LIKE 'v10_%_text_%') "
        "TO STDOUT WITH CSV HEADER"
    )
    cmd = ["psql", dsn, "-Atc", sql]
    try:
        with gzip.open(dest, "wb") as gz:
            proc = subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
            gz.write(proc.stdout)
    except subprocess.CalledProcessError as exc:
        if dest.exists():
            dest.unlink()
        return {
            "status": "error",
            "stderr": exc.stderr.decode("utf-8", errors="ignore")[-500:],
        }
    return {"status": "ok", "archive": str(dest), "size_bytes": dest.stat().st_size}


def _prune(backup_root: Path, retain: int) -> list[str]:
    snapshots = sorted(
        (p for p in backup_root.glob(f"{BACKUP_PREFIX}*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    deleted: list[str] = []
    for stale in snapshots[retain:]:
        shutil.rmtree(stale, ignore_errors=True)
        deleted.append(stale.name)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--retain", type=int, default=7, help="Keep the N most-recent durable snapshots."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, do not write.")
    args = parser.parse_args(argv)

    _load_env_file()

    object_root = Path(os.environ.get("QP__STORAGE__OBJECT_STORE_ROOT", str(DEFAULT_OBJECT_ROOT)))
    backup_root = Path(os.environ.get("QP_BACKUP_ROOT", str(DEFAULT_BACKUP_ROOT)))
    replay_source = object_root / REPLAY_CACHE_SUBPATH
    dsn = _pg_dsn_for_pg_dump()

    timestamp = _utc_timestamp()
    final_dir = backup_root / f"{BACKUP_PREFIX}{timestamp}"
    staging_dir = backup_root / f"{BACKUP_PREFIX}{timestamp}.partial"

    plan = {
        "timestamp_utc": timestamp,
        "object_root": str(object_root),
        "replay_source": str(replay_source),
        "replay_source_exists": replay_source.exists(),
        "postgres_dsn_present": dsn is not None,
        "destination": str(final_dir),
        "retain": args.retain,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if not replay_source.exists() and dsn is None:
        print(
            "ERROR: neither replay cache nor Postgres DSN available — nothing to back up.",
            file=sys.stderr,
        )
        return 2

    backup_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": (
            os.environ.get("COMPUTERNAME") or (os.uname().nodename if hasattr(os, "uname") else "")  # type: ignore[attr-defined]
        ),
        "object_root": str(object_root),
        "components": {},
    }

    manifest["components"]["replay_cache"] = _archive_replay_cache(
        replay_source, staging_dir / "replay_cache.tar.gz"
    )

    pg_status: list[dict[str, object]] = []
    if dsn:
        pg_status.append(_pg_dump_table(dsn, "text_events", staging_dir / "pg_text_events.sql.gz"))
        pg_status.append(
            _pg_dump_text_feature_vectors(dsn, staging_dir / "pg_feature_vectors_text.csv.gz")
        )
    manifest["components"]["postgres"] = pg_status

    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    staging_dir.rename(final_dir)
    pruned = _prune(backup_root, args.retain)

    summary = {
        "destination": str(final_dir),
        "components": manifest["components"],
        "pruned": pruned,
    }
    print(json.dumps(summary, indent=2))

    component_statuses = [manifest["components"]["replay_cache"].get("status")]  # type: ignore[union-attr]
    component_statuses.extend(item.get("status") for item in pg_status)
    if any(status == "error" for status in component_statuses):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
