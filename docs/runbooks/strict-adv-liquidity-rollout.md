# Strict ADV Liquidity Rollout

Use this when enabling stricter average-daily-volume liquidity gates.

## Why It Matters

ADV limits keep proposed orders within modeled liquidity. A strict rollout
should reduce order-size risk without unexpectedly blocking all targets.

## Configuration

Review liquidity settings in `.env` or the deployment secret/config source.
Confirm contracts include ADV fields when required.

## Rollout

1. Run in shadow mode and record blocked orders.
2. Run paper mode with conservative limits.
3. Compare proposed vs approved order sizes.
4. Review execution-quality and simulator-calibration artifacts.
5. Promote stricter settings only after readiness passes.

Commands:

```bash
python -m quant_platform run-engine --mode shadow --cycles 5
python -m quant_platform run-engine --mode paper --contracts-file ./contracts.json --cycles 1
python -m quant_platform readiness --help
```

## Rollback

- Restore prior liquidity settings.
- Keep blocked-order evidence.
- Re-run readiness.
- Do not bypass cash/risk/pre-trade gates to recover missed exposure.
