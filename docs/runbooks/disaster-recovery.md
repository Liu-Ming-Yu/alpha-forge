# Disaster recovery runbook

**Scope:** restore the paper-trading stack to a working state from the
artifacts produced by `scripts/backup_durable.py` (daily) and
`scripts/local_backup_drill.sh` (weekly).

**Recovery time objective (RTO):** 1 hour for the alpha stack;
3 hours including a full Postgres restore.

**Recovery point objective (RPO):** ≤ 24 h for LLM extractions,
≤ 24 h for Postgres feature vectors, ≤ 1 trading day for parquet
bar history (re-ingestible from Tiingo/Polygon).

> This is the *paper* DR runbook. There is no live capital at stake;
> the priority is rebuilding the **expensive-to-recompute** state
> (LLM cache, fitted manifests, model registry) before resuming the
> supervise loop.

---

## 1. Failure scenarios → which artifact you need

| Scenario                                                  | Restore from                                                            |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
| `docker compose down -v` wiped Postgres/Redis volumes     | `backups/durable_*/pg_text_events.sql.gz`, weekly drill `postgres.dump` |
| LLM extractions directory deleted / disk lost             | `backups/durable_*/replay_cache.tar.gz`                                 |
| Parquet bar store corrupted                               | Re-ingest from vendor (`data ingest --data-source tiingo`)              |
| Whole dev box lost                                        | Reclone repo + restore both backup tiers + re-promote model             |

---

## 2. Inventory: what's actually backed up

`scripts/backup_durable.py` writes `backups/durable_<UTC>/` daily:

```
manifest.json
replay_cache.tar.gz          # data/parquet/research/text_events/extractions/
pg_text_events.sql.gz        # text_events table (plain SQL, gzipped)
pg_feature_vectors_text.csv.gz  # text-family feature_vectors rows via COPY
```

`scripts/local_backup_drill.sh` writes `backups/<UTC>/` (weekly drill):

```
manifest.txt
postgres.dump                # custom-format full pg_dump
redis-save.txt               # marker; volume snapshot is operator-managed
object-store.tgz             # entire data/parquet/ tree
```

`scripts/backup_postgres.sh` writes `backups/scheduled/quant_<UTC>.dump`
(custom-format full pg_dump) on a separate cadence.

**Not backed up** (acceptable loss):

- Parquet bar history → re-ingestible via vendors in <1 hour for the
  300-name universe.
- Walk-forward campaign tearsheets → regeneratable from the campaign CLI.
- Redis streams payload → fail-closed engine state, recoverable from
  Postgres at next cycle.

---

## 3. Restore: full recovery sequence

Assumes you have the repo cloned and `.env` populated (see
`docs/runbooks/secrets-rotation.md` for `.env` reconstruction).

### Step 1 — pick the snapshot

```bash
ls -1t backups/ | head -10
LATEST="backups/$(ls -1t backups | grep ^durable_ | head -1)"
WEEKLY="backups/$(ls -1t backups | grep -E '^[0-9]{8}T' | head -1)"
echo "daily=$LATEST   weekly=$WEEKLY"
```

Read each manifest:

```bash
cat "$LATEST/manifest.json" | jq '.created_at_utc, .components'
cat "$WEEKLY/manifest.txt"
```

### Step 2 — bring up infra

```bash
docker compose up -d postgres redis
# wait for healthy:
docker compose ps
```

### Step 3 — restore Postgres

**A. From the weekly drill (preferred — full DB):**

```bash
gunzip -c "$WEEKLY/postgres.dump" 2>/dev/null \
  || pg_restore -d "$QP__STORAGE__POSTGRES_DSN" --clean --if-exists "$WEEKLY/postgres.dump"
```

**B. From the daily slice only (text tables, when no weekly drill exists):**

```bash
# Recreate schema first via Alembic — the daily slice is data-only:
.venv/bin/alembic upgrade head

zcat "$LATEST/pg_text_events.sql.gz" | psql "$QP__STORAGE__POSTGRES_DSN"

# feature_vectors text rows (CSV with header):
zcat "$LATEST/pg_feature_vectors_text.csv.gz" | \
  psql "$QP__STORAGE__POSTGRES_DSN" -c \
  "COPY feature_vectors FROM STDIN WITH CSV HEADER"
```

Verify:

```bash
psql "$QP__STORAGE__POSTGRES_DSN" -c "SELECT count(*) FROM text_events;"
psql "$QP__STORAGE__POSTGRES_DSN" -c \
  "SELECT count(*) FROM feature_vectors WHERE feature_id LIKE 'v10_%_text_%';"
```

### Step 4 — restore the LLM replay cache

```bash
mkdir -p data/parquet/research/text_events
tar -xzf "$LATEST/replay_cache.tar.gz" -C data/parquet/research/text_events/
ls data/parquet/research/text_events/extractions/deepseek/deepseek-v4-pro | head
```

### Step 5 — re-ingest bars (if the parquet bar store was lost)

```bash
python -m quant_platform data ingest \
  --data-source tiingo \
  --contracts-file infra/config/universe_300.json \
  --start 2022-01-01 --end "$(date -u +%F)"
```

### Step 6 — re-promote the model

The model registry lives in Postgres; a fresh DB needs the campaign
manifest re-registered:

```bash
python -m quant_platform governance production-candidate \
  --manifest data/parquet/research/walk_forward_durable_current_alpha_llm_risk_sized_v1/eligibility.json
python -m quant_platform governance readiness
```

### Step 7 — verify before restarting supervise

```bash
python -m quant_platform runtime smoke
python -m quant_platform runtime run-cycle --execution-backend ib-paper --once
# Only after both pass:
systemctl start quant-paper   # or the Windows PS wrapper
```

---

## 4. Partial restores

**Just need the LLM cache back** (most common — accidental rm):

```bash
tar -xzf "$LATEST/replay_cache.tar.gz" -C data/parquet/research/text_events/
```

The cache is content-addressable by request hash, so partial restores
are safe — the extraction pipeline simply re-fetches anything missing.

**Just need text_events for one ticker** (selective pg_restore):

```bash
zcat "$LATEST/pg_text_events.sql.gz" | \
  awk '/^COPY/,/^\\\.$/' | \
  grep -E '(^COPY|\\\.$|<TICKER_UUID>)' | \
  psql "$QP__STORAGE__POSTGRES_DSN"
```

---

## 5. Verification — tested-restore drill (quarterly)

The platform invariant is *backups exist iff the restore has been tested*.
Run this quarterly into a throwaway compose project:

```bash
# Spawn an isolated postgres on a different port. Pick any throwaway
# password; it never leaves the local docker network.
DR_TEST_PW="$(openssl rand -hex 8)"
docker run --rm -d --name pg-dr-test -e POSTGRES_PASSWORD="$DR_TEST_PW" -p 5433:5432 postgres:16

# Restore the most recent backup into it:
export TEST_DSN="postgresql://postgres:${DR_TEST_PW}@localhost:5433/postgres"
zcat backups/durable_$(ls backups | grep ^durable | tail -1 | sed 's/^durable_//')/pg_text_events.sql.gz \
  | psql "$TEST_DSN"

# Sanity check:
psql "$TEST_DSN" -c "SELECT count(*) FROM text_events;"

docker stop pg-dr-test
```

Record the drill outcome in `docs/runbooks/dr-drill-log.md` (create on
first run) with date, snapshot used, row count restored, and any
unexpected errors.

---

## 6. Known gotchas

- **Postgres 16 dumps are not restorable into 15.** Pin
  `image: postgres:16` in `docker-compose.yml` (already done).
- **Custom-format vs plain dumps** — the weekly drill uses
  `--format=custom` (needs `pg_restore`); the daily slice uses
  `--format=plain` + gzip (works with `psql`). Don't cross the streams.
- **The replay cache is hashed by full prompt+model+temperature.** If
  you restore a cache but bump `text_prompt_version`, all entries miss
  and you re-pay DeepSeek. Roll prompt versions only with a budget
  conversation first.
- **Volume wipes are the most common DR trigger** — `docker compose
  down -v` removed both `postgres_data` and `redis_data` on 2026-05-21.
  Aliased `dcdown` to `docker compose down` (no -v) on the dev box;
  do the same on any new operator host.
