# Cash Account Rules

This platform is cash-account first. Cash and settlement checks are core safety
rules, not convenience checks.

## Settlement

- Sell proceeds settle on T+1.
- Unsettled cash is tracked separately from settled cash.
- Buy-side settlement mirroring can be configured for parity, but order
  eligibility is still controlled by settled-cash rules.

## Reservations

Before a buy order is submitted, the platform reserves cash. Reservations are:

- Created before submission.
- Adjusted on partial fills.
- Released on cancel/reject/expiry.
- Deduplicated by order lifecycle identifiers.

## Buying Power

Available buying power is based on:

```text
settled_cash - reserved_cash - configured_cash_buffer
```

Risk policy can further reduce target size through single-name, sector, gross,
turnover, drawdown, and liquidity constraints.

## Prohibited Behavior

- No margin.
- No shorting.
- No using unsettled sell proceeds as settled buying power.
- No order submission while kill switch is active.
- No order submission with missing price or unmapped instrument identity.

## Recovery

If cash drift is detected:

1. Activate or keep the kill switch.
2. Sync broker account and positions.
3. Inspect reservations and settlement lots.
4. Reconcile broker-authoritative state.
5. Clear the kill switch only after operator review.
