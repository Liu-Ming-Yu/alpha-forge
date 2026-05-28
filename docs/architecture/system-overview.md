# System Overview

## Context

`quant-platform` runs a governed long-only equity strategy stack for one IBKR
cash account. It is designed for one operator, one deployable Python process,
and strict fail-closed behavior before any live order submission.

The architecture is a modular monolith. Service boundaries are enforced by
source layout, protocol contracts, import checks, service-coupling checks, and
tests instead of separate network services.

## Goals

- Maintain research-to-production parity across shadow, paper, and live paths.
- Enforce cash-account rules before any order leaves the platform.
- Persist evidence that matters for operator review and restart safety.
- Keep CLI/API entrypoints thin and move orchestration into application,
  bootstrap, and engine modules.
- Fail closed on stale data, missing contracts, schema mismatch, kill switch,
  cash/risk violations, and broker uncertainty.

## Non-Goals

- No margin, shorting, options, futures, or leveraged products in the current
  platform scope.
- No premature split into deployable microservices.
- No high-frequency trading or order-book replay engine.
- No multi-broker support.

## Service Map

```text
DataService
  owns ingest, bar stores, corporate actions, intraday validation, quorum

ResearchService
  owns features, audits, campaigns, backtests, reports, XGBoost, text signals
  (price-volume, fundamentals, formulaic, text-event-v2, microstructure-v3,
  ownership-v1, estimates-v1, options-v1, macro-v1, learned-representations-v1
  feature families register through FeatureRegistry; see
  docs/text-event-v2-family.md, docs/microstructure-v3-family.md,
  docs/ownership-v1-family.md, docs/estimates-v1-family.md,
  docs/options-v1-family.md, docs/macro-v1-family.md, and
  docs/learned-representations-v1-family.md for the most recently shipped
  families; docs/architecture/adr-002-learned-family-representation-choice.md
  records the PCA-over-alternatives decision for the learned family)

SignalService
  owns signal scoring and regime detection

PortfolioService
  owns cash ledger, settlement, risk policy, target construction, pre-trade gate

ExecutionService
  owns broker adapters, throttles, kill switch, order lifecycle, reconciliation

GovernanceService
  owns readiness, production candidates, paper-soak evidence, promotion gates

OperatorViews
  expose authenticated read-only API screens and health/status endpoints
```

Hot path:

```text
bars -> features -> signals -> target -> order intents -> pre-trade gate -> broker
```

## Storage

| Store | Purpose |
| --- | --- |
| PostgreSQL | Durable orders, positions, audit log, feature vectors, model registry, readiness, V2 state |
| Redis | Optional distributed lock and Redis Streams event bus |
| Parquet/object store | Historical bars and research artifacts |
| Local config files | Contracts, feature cards, feature families, observability examples |

## Runtime Entry Points

- `python -m quant_platform run-cycle`
- `python -m quant_platform supervise`
- `python -m quant_platform run-engine`
- `python -m quant_platform run-multi-engine`
- `python -m quant_platform serve-api`
- `python -m quant_platform preflight`
- `python -m quant_platform production-candidate`
- `python -m quant_platform smoke`

## Current Evidence State

The current code supports governed daily-bar, intraday, XGBoost, and text/LLM
research evidence. Evidence is only promotable when feature cards, audits,
forecast evidence, signal gates, production-candidate checks, and paper/live
readiness agree.

Daily OHLCV paper evidence is diagnostic until the relevant campaign, signal,
and production-candidate gates pass. V2 shared-account orchestration is wired as
the guarded multi-engine path and blocks competing live submitters when enabled.

## Technology

| Layer | Choice |
| --- | --- |
| Language | Python 3.11+ |
| Config | Pydantic settings with `QP__` env prefix |
| Database | SQLAlchemy async and psycopg 3 |
| Eventing/locks | Redis optional |
| Broker | IB Gateway/TWS through `ibapi` |
| API | FastAPI optional extra |
| Data | pandas, numpy, pyarrow |
| ML | XGBoost optional `ml` extra |
| LLM text | Anthropic SDK for shadow/gated text extraction |
| Quality | pytest, mypy, ruff, coverage, architecture ratchets |
