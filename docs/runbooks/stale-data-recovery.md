# Stale Data Recovery Runbook

Use this when bars, feature vectors, or dataset quorum evidence are stale.

## Identify

Check:

```bash
python -m quant_platform data-health --help
python -m quant_platform readiness --help
```

Review ingest logs, feature job status, object-store partitions, and feature
vector timestamps.

## Refresh Bars

```bash
python -m quant_platform ingest ^
  --start YYYY-MM-DD ^
  --end YYYY-MM-DD ^
  --contracts-file ./contracts.json
```

For intraday:

```bash
python -m quant_platform intraday --help
```

## Rebuild Features

```bash
python -m quant_platform features backfill --help
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
```

## Validate

```bash
python -m quant_platform data-health --help
python -m quant_platform readiness --help
python -m quant_platform dataset-quorum --help
```

## Resume

Resume only after:

- Bars are fresh.
- Feature vectors are fresh.
- Dataset quorum passes when required.
- Signals are no longer suppressed by stale-data gates.
- Operator has recorded the recovery decision.
