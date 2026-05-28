# Model Registry Promotions Runbook

Use this to promote, retire, roll back, or diff model registry entries.

## Preflight

Before registry changes:

- Confirm model artifact exists.
- Confirm manifest hash and feature schema.
- Confirm feature audits admit used features.
- Confirm campaign/signal evidence passes.
- Record the currently active model.

## Commands

```bash
python -m quant_platform model-registry --help
```

Common operations:

- List active and retired versions.
- Promote a version.
- Retire an active version.
- Roll back to a previous version.
- Diff two versions.

## Promotion

Promotion should be atomic and recorded in Postgres when a DSN is configured.
Do not promote from a local-only artifact without durable evidence.

## Rollback

1. Identify prior active version.
2. Retire or supersede the bad version.
3. Promote the prior version.
4. Re-run readiness/signal gate.
5. Record operator decision and evidence.

## Alerts To Watch

- Model preflight mismatch.
- Registry lookup failure.
- Manifest hash mismatch.
- Feature-schema mismatch.
- Prediction quality regression.
