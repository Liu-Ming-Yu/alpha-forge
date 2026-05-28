# Local Online Testing And Paper Soak

Use this for workstation testing against real Postgres/Redis and optional IBKR
paper connectivity.

## Workstation Profiles

Windows + WSL2/Docker talking to Windows-hosted TWS:

```bash
QP__BROKER__HOST=host.docker.internal
QP__BROKER__PORT=7497
```

macOS/Linux same namespace:

```bash
QP__BROKER__HOST=127.0.0.1
QP__BROKER__PORT=7497
```

## Required Environment

```bash
QP__STORAGE__POSTGRES_DSN=postgresql+psycopg://...
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__API__OPERATOR_API_KEY=<strong random key>
QP__BROKER__PAPER_TRADING=true
```

## Online Gate

```bash
python -m quant_platform verify-schema
python -m quant_platform health
python -m quant_platform ib-gateway-smoke --help
python -m quant_platform smoke --help
```

## Paper Soak

```bash
python -m quant_platform run-engine ^
  --mode paper ^
  --execution-backend ib-paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1

python -m quant_platform paper-soak --help
```

## Acceptance

- No unknown open orders.
- Broker positions reconcile.
- Data health passes.
- Feature/model/signal gates pass where required.
- Paper-soak artifact is recorded.
- Operator API health/readiness is authenticated and reachable.
