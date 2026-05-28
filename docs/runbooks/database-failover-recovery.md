# Database Failover Recovery Runbook

Use this when Postgres is unreachable, degraded, restored, or suspected to have
lost state.

## Immediate Action

1. Stop trading writers or activate the kill switch.
2. Preserve logs.
3. Check connectivity:

```bash
python -m quant_platform verify-schema
```

4. Check process/container health on the database host.

## Common Causes

- Postgres process down.
- Network path unavailable.
- Connection pool exhausted.
- Disk full.
- Migration failed or schema head mismatch.
- Credentials/DSN changed.

## Recovery

If Postgres restarts cleanly:

```bash
python -m quant_platform verify-schema
python -m quant_platform readiness --help
```

If restore is required:

```bash
scripts\restore_postgres.sh
python -m quant_platform verify-schema
```

## Resume Criteria

- Schema is at packaged head.
- Broker positions reconcile with internal state.
- Open orders are known.
- Kill switch review is complete.
- Readiness or smoke check passes.

## Post-Incident

Record:

- Incident timeline.
- Root cause.
- Backup/restore path if used.
- Schema revision before/after.
- Reconciliation result.
- Operator decision to resume.
