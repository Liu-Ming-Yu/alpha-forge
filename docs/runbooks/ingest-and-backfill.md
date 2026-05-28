# Ingest And Backfill Runbook

Use this for daily bar ingest, intraday import, corporate-action reprocessing,
and durable feature backfill.

## Prerequisites

- Contracts file.
- Object-store root.
- Vendor credentials when using Tiingo or Polygon.
- Postgres DSN when writing durable feature vectors or quorum evidence.

Optional vendor env:

```bash
QP__DATA_INGEST__TIINGO_API_TOKEN=<token>
QP__DATA_INGEST__POLYGON_API_KEY=<token>
QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN=["tiingo","polygon"]
```

## Daily Ingest

```bash
python -m quant_platform ingest ^
  --start YYYY-MM-DD ^
  --end YYYY-MM-DD ^
  --contracts-file ./contracts.json
```

## Maintenance Loop

```bash
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
```

## Corporate-Action Reprocess

```bash
python -m quant_platform reprocess-ca --help
```

Use when late corporate actions require adjusted partitions to be re-emitted.

## Durable Feature Backfill

```bash
python -m quant_platform features backfill ^
  --contracts-file ./contracts.json ^
  --start YYYY-MM-DDT00:00:00+00:00 ^
  --end YYYY-MM-DDT00:00:00+00:00 ^
  --feature-set-version paper-alpha-composite-v1 ^
  --date-policy nyse-sessions
```

## Intraday Utilities

```bash
python -m quant_platform intraday --help
```

Use intraday commands for vendor-neutral imports, validation, and quorum work.

## Daily Flow

1. Ingest end-of-day bars.
2. Verify coverage and stale-data logs.
3. Run maintenance/feature jobs.
4. Record dataset quorum if required.
5. Run readiness before paper/live sessions.

## Gap Recovery

If data is stale:

1. Backfill the affected date window.
2. Re-run feature jobs.
3. Confirm feature vectors are fresh.
4. Re-run data-health/readiness.
5. Resume only after stale-data gates pass.
