# Backup And Restore Runbook

Use this runbook for routine database backups, restore drills, and emergency
database recovery.

## Targets

| Target | Expectation |
| --- | --- |
| Postgres RPO | Latest scheduled backup or manual pre-change backup |
| Postgres RTO | Operator-defined; verify with local drill |
| Object store | Preserve bars and research artifacts separately |
| Redis | Treat locks/streams as operational state; Postgres remains authoritative for durable trading records |

## Backup

Run the project backup helper:

```bash
scripts\backup_postgres.sh
```

Record:

- Timestamp.
- Source DSN host/database.
- Backup path.
- File size.
- Schema revision.

## Restore

1. Stop engine and API writers.
2. Identify the backup.
3. Run the restore helper:

```bash
scripts\restore_postgres.sh
```

4. Verify schema:

```bash
python -m quant_platform verify-schema
```

5. Restart API/engine processes.
6. Run readiness or smoke:

```bash
python -m quant_platform smoke --help
python -m quant_platform readiness --help
```

## Weekly Drill

Use the local drill script:

```bash
scripts\local_backup_drill.sh
```

The drill should verify restore ability and row-count sanity without touching
production state.

## Incident Notes

If Postgres is corrupt or unreachable:

- Activate the kill switch.
- Stop writers.
- Preserve logs.
- Restore from the latest valid backup.
- Reconcile broker state before resuming trading.
