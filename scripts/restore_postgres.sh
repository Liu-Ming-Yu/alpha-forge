#!/usr/bin/env bash
# Restore Postgres from a pg_dump custom-format backup.
#
# Usage:
#   BACKUP_FILE=/mnt/backups/quant_20260501T120000Z.dump ./scripts/restore_postgres.sh
#
# WARNING: This script drops and recreates the quant_platform database.
#          It MUST NOT be run against a live production database without
#          operator sign-off. See docs/runbooks/backup-restore.md.
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

BACKUP_FILE="${BACKUP_FILE:-}"
if [ -z "$BACKUP_FILE" ]; then
  echo "ERROR: BACKUP_FILE is not set. Pass the path to the .dump file." >&2
  exit 2
fi
if [ ! -f "$BACKUP_FILE" ]; then
  echo "ERROR: backup file not found: $BACKUP_FILE" >&2
  exit 2
fi

# Strip psycopg driver prefix.
PG_DSN="${DSN/postgresql+psycopg:\/\//postgresql://}"

# Extract host/port/user/dbname from DSN for psql admin commands.
# Format: postgresql://user:pass@host:port/dbname
PG_HOST="$(echo "$PG_DSN" | sed -E 's|postgresql://[^@]+@([^:/]+).*|\1|')"
PG_PORT="$(echo "$PG_DSN" | sed -E 's|.*:([0-9]+)/.*|\1|')"
PG_USER="$(echo "$PG_DSN" | sed -E 's|postgresql://([^:@]+).*|\1|')"
PG_DB="$(echo "$PG_DSN" | sed -E 's|.*/([^?]+).*|\1|')"

if ! command -v pg_restore >/dev/null 2>&1; then
  echo "ERROR: pg_restore not found. Install postgresql-client." >&2
  exit 127
fi

echo "[restore] source:   $BACKUP_FILE"
echo "[restore] target:   $PG_DB @ $PG_HOST:$PG_PORT"
echo ""
read -rp "Type 'yes' to confirm drop + restore of '$PG_DB': " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Aborted."
  exit 1
fi

PSQL_ADMIN="psql -h $PG_HOST -p $PG_PORT -U $PG_USER"

echo "[restore] terminating existing connections..."
$PSQL_ADMIN -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$PG_DB' AND pid <> pg_backend_pid();" postgres

echo "[restore] dropping database..."
$PSQL_ADMIN -c "DROP DATABASE IF EXISTS $PG_DB;" postgres

echo "[restore] recreating database..."
$PSQL_ADMIN -c "CREATE DATABASE $PG_DB OWNER $PG_USER;" postgres

echo "[restore] restoring from $BACKUP_FILE..."
pg_restore --no-owner --role="$PG_USER" -d "$PG_DSN" "$BACKUP_FILE"

echo "[restore] verifying row counts..."
TABLES=("order_intents" "fill_events" "position_snapshots" "alembic_version")
for TABLE in "${TABLES[@]}"; do
  COUNT=$($PSQL_ADMIN -t -c "SELECT COUNT(*) FROM $TABLE;" "$PG_DSN" 2>/dev/null || echo "N/A")
  echo "  $TABLE: $COUNT"
done

echo "[restore] complete."
