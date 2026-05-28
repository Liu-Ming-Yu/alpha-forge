# Single VPS Industrial Readiness

Use this when running the platform on one durable host.

## Host Profile

Minimum expected services:

- Python runtime and project environment.
- PostgreSQL.
- Redis.
- IBKR TWS/Gateway if broker-backed paper/live.
- Process manager or service runner.
- Log collection.
- Backup job.
- Metrics scrape target.

## Process Layout

Run separate processes for:

- Operator API.
- Data maintenance.
- Engine/supervisor.
- Backup/restore drill scheduling.
- Observability scrape/alerts.

## Readiness

Before paper/live:

```bash
python -m quant_platform verify-schema
python -m quant_platform smoke --help
python -m quant_platform readiness --help
python -m quant_platform production-candidate --help
```

## Paper Soak Evidence

Run paper cycles and then:

```bash
python -m quant_platform paper-soak --help
```

Evidence should include broker health, data health, lifecycle state,
reconciliation, latency, prediction quality, simulator calibration, and operator
API readiness.

## Failure Posture

Default to halt:

- Broker uncertain.
- Database unavailable.
- Redis lock unavailable when required.
- Stale data.
- Missing contracts.
- Failed readiness gates.
