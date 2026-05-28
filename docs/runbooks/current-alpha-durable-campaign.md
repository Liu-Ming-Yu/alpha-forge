# Current Alpha Durable Campaign Runbook

Use this runbook to rerun or inspect the current durable alpha campaign from
Postgres feature vectors and tracked feature-card metadata.

## Required Inputs

- Contracts file.
- Feature-set version.
- Feature-card directory.
- Feature-family file when applicable.
- Durable feature vectors in Postgres.
- Object-store root for campaign artifacts.
- Model version.
- Date policy and horizon.

## Populate Durable Features

```bash
python -m quant_platform features backfill ^
  --contracts-file infra/config/backtest_contracts.json ^
  --start YYYY-MM-DDT00:00:00+00:00 ^
  --end YYYY-MM-DDT00:00:00+00:00 ^
  --feature-set-version paper-alpha-composite-v1 ^
  --bar-seconds 86400 ^
  --date-policy nyse-sessions
```

## Attribute Failures

```bash
python -m quant_platform research-campaign attribute-feature-failures --help
```

Use this before promotion to explain quarantined or zero-coverage features.

## Run Campaign

```bash
python -m quant_platform research-campaign run --help
```

Review the generated `campaign_manifest.json` and supporting artifacts.

## Promotion Decision

Campaign output is not promotion evidence unless:

- Feature audits admit the used features.
- Walk-forward and paper metrics pass thresholds.
- Slippage-adjusted metrics pass the configured gate.
- Signal gate passes.
- Readiness and production-candidate gates pass.
- The output declares the next allowed mode.

## Failure Handling

If campaign eligibility fails:

- Keep the package in shadow-only mode.
- Preserve artifacts.
- Record blocked features and metrics.
- Do not register or promote the model as paper/live evidence.
