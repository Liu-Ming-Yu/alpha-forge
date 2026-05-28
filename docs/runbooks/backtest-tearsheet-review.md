# Backtest Tearsheet Review Runbook

Use this runbook to inspect a completed backtest artifact before it becomes
promotion evidence.

## Generate

```bash
python -m quant_platform tearsheet --help
python -m quant_platform backtest --help
```

Expected artifact families:

- `metrics.json`: return, drawdown, turnover, and headline risk.
- `execution_quality.json`: fills, ADV participation, modeled costs, slippage.
- `ic_report.json`: information coefficient and decay.
- `attribution.json`: source, factor, sector, or regime contribution.
- `run_summary.json`: strategy/run metadata.

## Review Order

1. Confirm universe, horizon, date policy, and feature-set version.
2. Confirm train/test or walk-forward split has no lookahead leakage.
3. Confirm feature audits admit every feature used by the model.
4. Review drawdown, turnover, and gross exposure.
5. Review execution-quality assumptions and simulator calibration.
6. Review IC stability, sign consistency, and decay.
7. Confirm artifacts are comparable to the promotion target.

## Promotion Red Flags

- Missing feature-card or audit evidence.
- Comparing runs with different universes, horizons, or date policies.
- Sharpe or drawdown computed from too few observations.
- Negative IC for a feature whose expected sign was positive.
- Turnover above the paper/live execution budget.
- ADV participation above configured limits.
- Fill model not calibrated against paper fills.

## Acceptance

Backtest evidence alone cannot promote a model. It must be paired with feature
admission, signal gate evidence, paper evidence, readiness, and
production-candidate checks.
