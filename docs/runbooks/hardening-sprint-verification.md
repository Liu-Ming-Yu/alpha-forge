# Hardening Sprint Verification

This document records the verification shape used for hardening work. Keep it
focused on reproducible gates rather than historical prose.

## Offline Gate

```bash
make verify
```

The script recreates `.venv-verify`, installs constrained dependencies, and
runs:

- `git diff --check`
- Ruff lint and format checks.
- Generated-artifact cleanup/check.
- Import-boundary and composition ratchets.
- Service-coupling ratchets.
- Module-size, type-debt, and lint-debt ratchets.
- Secret scan.
- `mypy src`.
- Application coverage subset.
- Broad offline pytest suite with coverage floor.

## Optional Gates

Durable Postgres/Redis:

```bash
set QP_VERIFY_DURABLE=1
make verify
```

Live IBKR:

```bash
set QP_VERIFY_LIVE_IBKR=1
set IBAPI_PACKAGE_PATH=<path-to-TWS-API/source/pythonclient>
make verify
```

## Acceptance Record

For each sprint or production-hardening pass, record:

- Branch or commit.
- Commands run.
- Pass/fail status.
- Skipped optional gates and why.
- Remaining risks or follow-up tickets.

## Red Lines

- No captured pytest output may contain a `FAILED` summary for acceptance.
- Marker-only test subsets are diagnostics, not acceptance.
- Stale docs, failed ratchets, or generated artifacts should be fixed before
  requesting merge.
