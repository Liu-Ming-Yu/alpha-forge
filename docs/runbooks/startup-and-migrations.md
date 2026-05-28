# Startup And Migrations Runbook

Use this for packaged Alembic migration checks and startup schema validation.

## Schema Ownership

The canonical migration tree lives under:

```text
src/quant_platform/alembic
```

The packaged application should not depend on a root-level migration tree.

## Check Migration Chain

```bash
python -m quant_platform migrations-check
```

## Apply Migrations

```bash
python -m quant_platform migrate
```

This reads `QP__STORAGE__POSTGRES_DSN`.

## Verify Startup Schema

```bash
python -m quant_platform verify-schema
```

Postgres-backed startup paths fail closed if the database is not at the packaged
head.

## Common Failures

| Failure | Meaning | Action |
| --- | --- | --- |
| Missing `alembic_version` | Database was not initialized | Run `migrate` after backup/review |
| DB behind packaged head | Pending migrations | Run `migrate` |
| DB ahead of packaged head | Binary is older than database | Deploy matching application version |
| Migration asset missing | Packaging/deployment issue | Redeploy with packaged Alembic resources |

## Unauthenticated API Mode

The operator API refuses unauthenticated protected routes unless both explicit
risk flags are set:

```bash
QP__API__ALLOW_UNAUTHENTICATED=true
QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true
```

Use only for isolated local development.
