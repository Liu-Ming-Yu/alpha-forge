# Alpha Paper Enablement Runbook

Use this runbook when moving a governed alpha package from diagnostic shadow
evidence into paper operation.

## Preconditions

- Feature cards exist for every candidate feature.
- Feature audits pass at the required state.
- The model artifact has a manifest and registry entry.
- Signal gate evidence passes for the target horizon.
- Dataset quorum evidence is current when required.
- `readiness`, `paper-soak`, and `production-candidate` commands are available.

## Environment

Required basics:

```bash
QP__STORAGE__POSTGRES_DSN=postgresql+psycopg://...
QP__STORAGE__REDIS_URL=redis://...
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__API__OPERATOR_API_KEY=<strong random key>
QP__BROKER__PAPER_TRADING=true
```

For IBKR paper:

```bash
QP__BROKER__HOST=127.0.0.1
QP__BROKER__PORT=7497
```

## Enablement Sequence

1. Verify schema and migrations:

```bash
python -m quant_platform migrations-check
python -m quant_platform verify-schema
```

2. Check feature admission:

```bash
python -m quant_platform features audit status --help
python -m quant_platform features audit assert --help
```

3. Check signal and readiness gates:

```bash
python -m quant_platform signal-gate --help
python -m quant_platform readiness --help
python -m quant_platform production-candidate --help
```

4. Run paper engine cycle:

```bash
python -m quant_platform run-engine ^
  --mode paper ^
  --execution-backend ib-paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

5. Generate paper-soak evidence:

```bash
python -m quant_platform paper-soak --help
```

## Rollback

- Disable the alpha source weight or set the engine back to shadow mode.
- Clear any pending orders only through the broker-aware cancellation path.
- Keep the model artifact and failed evidence for audit history.
- Re-run readiness after rollback to confirm no paper/live state remains stale.
