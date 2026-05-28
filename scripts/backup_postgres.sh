#!/usr/bin/env bash
# Scheduled Postgres backup — designed for cron/systemd timer.
#
# Usage:
#   POSTGRES_DSN="postgresql+psycopg://..." BACKUP_DIR="/mnt/backups" ./scripts/backup_postgres.sh
#
# Environment variables:
#   QP__STORAGE__POSTGRES_DSN  — platform DSN (postgresql+psycopg:// or postgresql://)
#   BACKUP_DIR                 — destination directory (default: ./backups/scheduled)
#   BACKUP_RETAIN_DAYS         — days to keep backups (default: 30)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

DSN="${QP__STORAGE__POSTGRES_DSN:-}"
if [ -z "$DSN" ]; then
  echo "ERROR: QP__STORAGE__POSTGRES_DSN is not set." >&2
  exit 2
fi

# Strip the psycopg driver prefix so pg_dump can use it.
PG_DSN="${DSN/postgresql+psycopg:\/\//postgresql://}"

BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups/scheduled}"
RETAIN_DAYS="${BACKUP_RETAIN_DAYS:-30}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$BACKUP_DIR/quant_${TIMESTAMP}.dump"

mkdir -p "$BACKUP_DIR"

if ! command -v pg_dump >/dev/null 2>&1; then
  echo "ERROR: pg_dump not found. Install postgresql-client." >&2
  exit 127
fi

echo "[backup] writing $DEST"
pg_dump --format=custom --file "$DEST" "$PG_DSN"
echo "[backup] done — $(du -sh "$DEST" | cut -f1)"

# Prune backups older than RETAIN_DAYS.
find "$BACKUP_DIR" -name "quant_*.dump" -mtime +"$RETAIN_DAYS" -delete
echo "[backup] pruned backups older than ${RETAIN_DAYS} days"
