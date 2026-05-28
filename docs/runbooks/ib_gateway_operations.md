# IB Gateway Operations Runbook

Use this for IBKR paper/live connectivity, contract identity, health checks,
and recovery.

## Prerequisites

- IBKR TWS or IB Gateway running.
- API socket enabled.
- Correct paper/live port.
- `ibapi` installed from the IBKR TWS API distribution.
- Contracts file with canonical broker identifiers where required.

## Ports

| App | Paper | Live |
| --- | ---: | ---: |
| TWS | `7497` | `7496` |
| IB Gateway | `4002` | `4001` |

## Environment

```bash
QP__BROKER__HOST=127.0.0.1
QP__BROKER__PORT=7497
QP__BROKER__PAPER_TRADING=true
```

For WSL2/Docker talking to Windows-hosted TWS, use:

```bash
QP__BROKER__HOST=host.docker.internal
```

## Smoke Checks

```bash
python -m quant_platform health
python -m quant_platform ib-gateway-smoke --help
python -m quant_platform ib-paper-lifecycle --help
```

The paper lifecycle check can place paper orders when explicitly configured.
Do not run it against a live account.

## Contract Identity

Live and IBKR paper paths require a contracts file. Unmapped broker conIds,
positions, fills, or open orders must fail closed and require operator review.

## Paper Run

```bash
python -m quant_platform run-engine ^
  --mode paper ^
  --execution-backend ib-paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

## Kill Switch

The kill switch prevents new order submissions. Keep it active during broker
uncertainty, reconnects, and reconciliation discrepancies. Clear it only after
operator review and successful recovery checks.

## Reconciliation

After reconnect or manual broker action:

1. Sync account and positions.
2. Inspect open orders.
3. Map every broker position/fill to a known instrument.
4. Resolve operator-required discrepancies.
5. Record the decision before resuming.

## Common IBKR Failure Modes

- API socket disabled.
- Wrong paper/live port.
- Client ID already in use.
- Gateway/TWS logged out or stuck after maintenance.
- Missing contract conId.
- Position/fill cannot be mapped to instrument ID.
- Order remains uncertain after disconnect.
