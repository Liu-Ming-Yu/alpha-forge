# Kill-Switch Recovery Runbook

Use this whenever the kill switch is active, whether it was triggered manually
or automatically.

## Identify Cause

Common causes:

- Drawdown breach.
- Cash drift.
- Reconciliation discrepancy.
- Broker uncertainty.
- Cycle error.
- Manual operator halt.

Inspect health/readiness:

```bash
python -m quant_platform health
python -m quant_platform readiness --help
```

## Recovery Checks

Before clearing:

- Broker is connected.
- Open orders are known or zero.
- Broker positions reconcile to internal state.
- Cash drift is within tolerance.
- No stale data gate is blocking signals.
- The original trigger is understood.

The engine loop performs read-only recovery checks while blocked and reports
whether state appears ready for operator clear.

## Clear

Clear only through the supported operator path or CLI/API endpoint for the
current deployment. Record:

- Operator ID.
- Reason for clear.
- Evidence reviewed.
- Timestamp.

## Resume

After clearing:

1. Run health/readiness.
2. Resume with a bounded paper or shadow cycle first when possible.
3. Watch broker and reconciliation logs.
4. Confirm kill switch remains clear.

## Do Not

- Clear while open orders are uncertain.
- Clear before reconciling broker state after manual action.
- Clear because a scheduled cycle is waiting.
- Bypass cash/risk/execution policy to "catch up".
