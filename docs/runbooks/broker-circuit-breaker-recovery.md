# Broker Circuit-Breaker Recovery Runbook

Use this when broker health, reconnect, lifecycle sync, or order uncertainty
causes a halt.

## Triage

Check broker health:

```bash
python -m quant_platform health
python -m quant_platform ib-gateway-smoke --help
```

Review:

- Gateway/TWS process status.
- Recent platform logs.
- Open orders.
- Broker account and position sync.
- Kill-switch state.

## Wait Or Restart

Wait briefly when latency is elevated but the broker remains connected and no
orders are uncertain. Restart when the gateway is disconnected, callbacks stop,
or account/position sync cannot complete.

## Restart IB Gateway/TWS

1. Stop the platform engine.
2. Confirm no new orders are being submitted.
3. Restart TWS/IB Gateway.
4. Wait for API connectivity.
5. Run:

```bash
python -m quant_platform ib-gateway-smoke --help
```

## Reconcile

After reconnect:

```bash
python -m quant_platform health
python -m quant_platform readiness --help
```

Confirm:

- Broker connected.
- Open orders are known.
- Positions map to configured contracts.
- Reconciliation has no operator-required discrepancies.
- Kill switch is clear only after operator review.

## Escalation

If gateway recovery exceeds the operator time budget:

- Keep the kill switch active.
- Cancel broker-side orders manually only if the platform cannot do so.
- Record manual broker actions.
- Reconcile before resuming.
