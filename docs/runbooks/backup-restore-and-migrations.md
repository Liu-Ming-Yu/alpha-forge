# Backup, Restore, And Migration Rollback

Use this short runbook when database recovery and schema changes are involved in
the same incident or release.

## Before Migration

1. Stop writers or move them to a safe maintenance state.
2. Confirm the current schema head:

```bash
python -m quant_platform verify-schema
```

3. Take a database backup:

```bash
scripts\backup_postgres.sh
```

4. Record the backup path and current Alembic revision.

## Apply Migration

```bash
python -m quant_platform migrate
python -m quant_platform verify-schema
```

Then run the relevant smoke or readiness command:

```bash
python -m quant_platform smoke --help
python -m quant_platform readiness --help
```

## Rollback

Prefer restoring from the pre-migration backup. Alembic downgrades should only
be used after reviewing the migration's `downgrade()` function and confirming
data impact.

Restore sequence:

1. Stop engine/API writers.
2. Restore the backup.
3. Run `verify-schema`.
4. Re-run broker/account reconciliation if trading state is involved.
5. Restart services.

## Evidence To Keep

- Backup file path.
- Pre/post Alembic revisions.
- Restore command output.
- Smoke/readiness output.
- Operator decision log.
