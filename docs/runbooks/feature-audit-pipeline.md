# Feature Audit Pipeline Runbook

Use this runbook to admit, quarantine, assert, or retire governed features.

## Principle

No governed feature should influence model training, scoring, comparison, or
promotion evidence unless its feature card and latest audit admit it at the
required state.

## Feature Card

A feature card should declare:

- Feature name and version.
- Owner.
- Economic thesis.
- Source datasets.
- Required lags and point-in-time assumptions.
- Expected sign and horizon.
- Valid universe and turnover expectations.
- Risk exposures.
- Failure modes.
- Current state.

## Run Audit

```bash
python -m quant_platform features audit run --help
```

Typical inputs:

- `--feature-card`
- `--samples`
- `--feature-set-version`
- `--horizon-days`
- `--baseline-features`
- `--persist`

## Inspect And Assert

```bash
python -m quant_platform features audit status --help
python -m quant_platform features audit assert --help
python -m quant_platform features audit retire --help
```

Use `assert` before model training or promotion to fail closed when a feature is
not admitted.

## Artifacts

Audit artifacts are written below:

```text
$QP__STORAGE__OBJECT_STORE_ROOT/research/feature_audits/<feature>/<version>/<audit_id>/
```

Expected artifacts include the decision, metrics, blocked reasons, and feature
card hash.

## Campaign Integration

Research campaigns must:

- Load admitted feature lists.
- Exclude quarantined features from training and inference.
- Record blocked feature summaries.
- Fail closed if required feature-card hashes do not match.

## Operator Evidence

Feature audit status is surfaced through CLI and operator API routes. Use the
operator view for read-only review; use CLI commands for state changes.
