# Runbook — Alpha Improvement & Live-Readiness Rollout

This runbook executes the alpha-improvement workstreams end to end. The code for
all seven workstreams is merged on branch `alpha-improvement-live-ready`; the
steps below are the **operator-run, infrastructure-dependent** half — they need
network access (Polygon), Postgres + Redis, and (for the final step) IB paper.

**Acceptance gate:** the final walk-forward campaign `eligibility.json` reads
`"passed": true` with `slippage_adjusted_sharpe >= 1.0`.

## Prerequisites

- `.env` has `QP__DATA_INGEST__POLYGON_API_KEY` and `QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN=["polygon"]` (already set).
- Postgres and Redis running (`docker compose up -d postgres redis`); `QP__STORAGE__*` point at them.
- A second EOD vendor (Tiingo) is needed for the 2-vendor dataset quorum — set
  `QP__DATA_INGEST__TIINGO_API_TOKEN` and `BAR_FETCH_FALLBACK_CHAIN=["polygon","tiingo"]` if quorum is required for live.

## 1 — Build the universe & ingest data (WS1)

```bash
# Build a ~300-name liquid US-equity contracts file from Polygon reference data.
python scripts/build_universe.py --out infra/config/universe_300.json --count 300

# Vendor-primary historical ingest (skips IB pacing limits).
python -m quant_platform ingest --data-source vendor \
  --contracts-file infra/config/universe_300.json \
  --start 2022-01-01 --end 2026-05-01 --bar-seconds 86400

# Compute + backfill features at the new feature-set version 1.1.0.
python -m quant_platform features backfill \
  --contracts-file infra/config/universe_300.json \
  --start 2022-06-01 --end 2026-05-01 --feature-set-version 1.1.0
python -m quant_platform features build-samples \
  --contracts-file infra/config/universe_300.json \
  --feature-set-version 1.1.0 --horizon-days 21 \
  --output data/parquet/research/_inputs/u300_samples.json
```

Verify every instrument has a contiguous 2022–2026 bar series before continuing.
`build_universe.py` emits `con_id: 0` — resolve real IB con_ids with a one-time
`reqContractDetails` pass before any live/paper order routing (step 5).

## 2 — Fit factor weights + cost-aware sweep (WS3 + WS4 + WS5)

The new diversifying factors (`reversal_21d`, `low_volatility_63d`,
`mean_reversion_63d`) are already in the `close` family at feature-set `1.1.0`.

```bash
# Sweep no-trade band x rebalance interval to lift slippage-adjusted Sharpe.
for band in 0.002 0.005 0.01; do
  for interval in 1 5 10; do
    python -m quant_platform research-campaign run \
      --contracts-file infra/config/universe_300.json \
      --start 2022-06-01 --end 2026-05-01 \
      --feature-set-version 1.1.0 --model-version u300-classical-v1 \
      --signal-type classical --feature-audit-mode paper \
      --campaign-top-n 40 \
      --campaign-no-trade-band $band \
      --campaign-rebalance-interval-days $interval
  done
done
```

Pick the run whose `eligibility.json` shows `slippage_adjusted_sharpe >= 1.0`
with `oos_rolling_ic` still `>= 0.05`. The campaign fits IC-weighted factor
weights into `selected_weights`; the six-gate feature audit (now enforcing
rank-normalization) admits only stable factors.

## 3 — Wire the fitted weights into the live model (WS3)

Point the live classical model at the winning campaign manifest:

```bash
# In .env:
QP__FACTORS__FITTED_WEIGHTS_MANIFEST=data/parquet/research/walk_forward/<run-id>/campaign_manifest.json
```

The live `LinearWeightSignalModel` now uses the data-fitted weights and is
pinned to feature-set `1.1.0` (stale rebuilds fail closed).

## 4 — Promote gated non-classical sources (WS6)

Per source (`xgboost`, `text`, `event`, `intraday`):

```bash
python -m quant_platform research-campaign run \
  --contracts-file infra/config/universe_300.json \
  --start 2022-06-01 --end 2026-05-01 \
  --signal-type <source> --feature-audit-mode paper [--train-xgboost]
python -m quant_platform signal-gate assert ...
python -m quant_platform production-candidate ...
```

Then wire artifacts in `.env` (`QP__BOOSTING__ARTIFACT_MANIFEST`,
`QP__ALPHA__SOURCE_WEIGHTS`, `QP__ALPHA__ENSEMBLE_MODE=paper`) and ramp
`QP__ALPHA__PAPER_MAX_NON_CLASSICAL_WEIGHT` from 0.10 upward as evidence
accrues. Keep a source at weight 0 until its `eligibility.json` passes.

## 5 — Operate & revalidate (WS7)

```bash
# Re-promote the classical model after the volume wipe.
python -m quant_platform production-candidate ...
python -m quant_platform readiness ...

# Smoke, then run the continuous paper loop against IB paper.
python -m quant_platform smoke
python -m quant_platform supervise --engine cross_sectional_equity \
  --mode paper --execution-backend ib-paper \
  --contracts-file infra/config/universe_300.json --interval 300 --max-cycles 5
```

**Final acceptance:** re-run the step-2 winning campaign on the fully
backfilled 300-name data → `eligibility.json` `"passed": true`. Only then
promote toward live.
