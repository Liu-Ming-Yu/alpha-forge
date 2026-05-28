#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

STAMP="${QP_BACKUP_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BACKUP_ROOT="${QP_BACKUP_ROOT:-$ROOT/backups}"
DEST="$BACKUP_ROOT/$STAMP"
OBJECT_ROOT="${QP__STORAGE__OBJECT_STORE_ROOT:-$ROOT/data/parquet}"

mkdir -p "$DEST"

if [ -z "${QP__STORAGE__POSTGRES_DSN:-}" ]; then
  echo "QP__STORAGE__POSTGRES_DSN is required for the Postgres backup." >&2
  exit 2
fi
if ! command -v pg_dump >/dev/null 2>&1; then
  echo "pg_dump is required. Install postgresql-client inside WSL2 before running the backup drill." >&2
  exit 127
fi

echo "Writing backup drill artifacts to $DEST"
pg_dump --format=custom --file "$DEST/postgres.dump" "$QP__STORAGE__POSTGRES_DSN"

if command -v redis-cli >/dev/null 2>&1 && [ -n "${QP__STORAGE__REDIS_URL:-}" ]; then
  redis-cli -u "$QP__STORAGE__REDIS_URL" ping >/dev/null
  redis-cli -u "$QP__STORAGE__REDIS_URL" save >/dev/null
  echo "redis SAVE completed" > "$DEST/redis-save.txt"
else
  echo "redis-cli or QP__STORAGE__REDIS_URL unavailable; Redis volume snapshot is operator-managed." \
    > "$DEST/redis-save.txt"
fi

if [ -d "$OBJECT_ROOT" ]; then
  tar -C "$OBJECT_ROOT" -czf "$DEST/object-store.tgz" .
else
  echo "object store root not found: $OBJECT_ROOT" > "$DEST/object-store-missing.txt"
fi

cat > "$DEST/manifest.txt" <<EOF
backup_stamp=$STAMP
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
postgres_dump=$DEST/postgres.dump
redis_marker=$DEST/redis-save.txt
object_store_root=$OBJECT_ROOT
object_store_archive=$DEST/object-store.tgz
EOF

echo "Backup drill complete: $DEST"
