# Production Deployment Runbook

Use this as the release checklist for durable paper or live deployment.

## Package Install

```bash
python -m pip install -e ".[api,ml]"
```

Install `ibapi` from the IBKR TWS API distribution when using IBKR.

## Required Infrastructure

- PostgreSQL.
- Redis when using locks or Redis Streams.
- Object-store path for bars and artifacts.
- IBKR TWS/Gateway for broker-backed paper/live.
- Operator API key.
- Observability stack if exposing `/metrics`.

## Migration

```bash
python -m quant_platform migrations-check
python -m quant_platform migrate
python -m quant_platform verify-schema
```

## Runtime Processes

Common processes:

- Engine or supervisor process.
- Operator API process.
- Data maintenance process.
- Observability scrape target.

Example:

```bash
python -m quant_platform serve-api --host 127.0.0.1 --port 8000
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
python -m quant_platform run-engine --mode paper --contracts-file ./contracts.json --cycles 1
```

## Release Gates

Before paper/live promotion:

```bash
python -m quant_platform preflight --help
python -m quant_platform readiness --help
python -m quant_platform production-candidate --help
python -m quant_platform smoke --help
```

Live requires additional operator review and current broker readiness.

## Rollback

- Stop writers.
- Activate or keep the kill switch.
- Revert config or model registry state.
- Restore database if migration/data state caused the failure.
- Verify schema, health, readiness, and reconciliation before resume.
