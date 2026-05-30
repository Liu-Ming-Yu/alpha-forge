# Documentation Index

This directory contains architecture notes, interface contracts, and operational
runbooks for `quant-platform`.

## Start Here

- [Root README](../README.md): project shape, setup, command examples.
- [USEME](../USEME.md): operator/developer quickstart.
- [Context glossary](../CONTEXT.md): project vocabulary.
- [Contributing](../CONTRIBUTING.md): CI and architecture rules.

## Architecture

- [System overview](architecture/system-overview.md): current modular-monolith
  map.
- [Service boundaries](architecture/service-boundaries.md): dependency rules and
  ratchets.
- [ADR-002 — Learned family representation choice](architecture/adr-002-learned-family-representation-choice.md):
  why PCA was selected over autoencoder, MoE, XGBoost leaf indices,
  and explicit interaction expansion for the `learned` family.
- [Production roadmap](architecture/production-roadmap.md): completed phases,
  current phase, deferred scope.
- [Risk register](architecture/risk-register.md): active and retired safety
  risks.
- [Source audit checklist](architecture/source-audit-checklist.md): review
  prompts for production-hardening work.
- [V2 execution flow](architecture/v2-execution-flow.md): multi-engine account
  orchestration path.
- [ADR-014 — Unified engine runtime](architecture/adr-014-unified-engine-runtime.md):
  single-engine `supervise` runs as the N=1 case of the multi-engine
  orchestrator so the operator console reflects whatever runs.
- [Core contracts](interfaces/core-contracts.md): contract groups and invariant
  expectations.

## Runbooks

Operations:

- [Production deployment](runbooks/production-deployment.md)
- [Startup and migrations](runbooks/startup-and-migrations.md)
- [Local online testing](runbooks/local-online-testing.md)
- [Single VPS industrial readiness](runbooks/single-vps-industrial-readiness.md)
- [Observability and Alertmanager](runbooks/observability-alertmanager.md)
- [Secrets and artifacts](runbooks/secrets-and-artifacts.md)

Data and research:

- [Ingest and backfill](runbooks/ingest-and-backfill.md)
- [Strict ADV liquidity rollout](runbooks/strict-adv-liquidity-rollout.md)
- [Feature audit pipeline](runbooks/feature-audit-pipeline.md)
- [Current alpha durable campaign](runbooks/current-alpha-durable-campaign.md)
- [LLM text feature promotion](runbooks/llm-text-feature-promotion.md)
- [Backtest tearsheet review](runbooks/backtest-tearsheet-review.md)
- [Model registry promotions](runbooks/model-registry-promotions.md)
- [Alpha paper enablement](runbooks/alpha-paper-enablement.md)

Feature families:

- [`microstructure-v3` family reference](microstructure-v3-family.md) — 19
  daily-OHLCV-derived microstructure proxies (range-based vols including
  Yang-Zhang, Roll + Corwin-Schultz spread estimators, close-in-range,
  serial dependence, volume-return coupling, range asymmetry, bipower
  variation, realized skew/kurt, Lo-MacKinlay variance ratio, range
  persistence, Median Realized Variance, tripower variation, realized
  jump intensity).
- [`ownership-v1` family reference](ownership-v1-family.md) — 6 features
  from 13F institutional holdings + FINRA short-interest snapshots.
  **Scaffold only**: real vendor data feeds (Sharadar SF3, FINRA SI
  files) not yet wired.
- [`estimates-v1` family reference](estimates-v1-family.md) — 6 features
  from analyst consensus snapshots + historical earnings-surprise
  records (revision magnitude, direction, dispersion, coverage,
  past-4-quarter surprise mean, revenue revision). **Scaffold only**:
  real vendor feed (IBES, FactSet, Visible Alpha) not yet wired.
- [`options-v1` family reference](options-v1-family.md) — 6 features
  from daily options-implied snapshots (ATM IV, 25Δ skew, IV term
  slope, put/call volume + OI ratios, IV-vs-realized premium).
  **Scaffold only**: real options-chain vendor (CBOE, OptionMetrics,
  Polygon options, ORATS) not yet wired.
- [`macro-v1` family reference](macro-v1-family.md) — 6 macro/regime
  features from 8 FRED time series (yield-curve slopes, credit
  spread, VIX, dollar momentum, real yield). **Feed-agnostic**:
  family takes `MacroSeriesValue` records; an optional
  `fetch_fred_series()` helper wraps the free FRED API for the
  common case.
- [`learned-representations-v1` family reference](learned-representations-v1-family.md)
  — 9 features from an artifact-backed PCA transform of the other
  9 families' outputs (8 PC scores + 1 reconstruction error). The
  family is **representation-only**: fitting belongs to the trainer
  (lazy sklearn import in `learned/trainer.py`); compute is a pure
  deterministic matmul. See [ADR-002](architecture/adr-002-learned-family-representation-choice.md)
  for why PCA was chosen over autoencoder / MoE / XGBoost leaves /
  pairwise expansion.
- [`text-event-v2` family reference](text-event-v2-family.md) — 27
  LLM-extracted features across news + filings + earnings calls.
- [Text/event alpha ingestion scope](text-event-alpha-scope.md) — the
  separate SEC EDGAR + DeepSeek pipeline scoped for paper-alpha campaigns.

Incident response:

- [Kill-switch recovery](runbooks/kill-switch-recovery.md)
- [Broker circuit-breaker recovery](runbooks/broker-circuit-breaker-recovery.md)
- [Stale data recovery](runbooks/stale-data-recovery.md)
- [Database failover recovery](runbooks/database-failover-recovery.md)
- [Backup and restore](runbooks/backup-restore.md)
- [Backup, restore, and migration rollback](runbooks/backup-restore-and-migrations.md)
- [Distributed-lock operations](runbooks/distributed-lock-operations.md)
- [Redis Streams event bus](runbooks/redis-streams-event-bus.md)
- [Event-bus retention](runbooks/event-bus-retention.md)
- [Cash-account rules](runbooks/cash-account-rules.md)
- [Operator API key lifecycle](runbooks/operator_rbac.md)
- [Hardening sprint verification](runbooks/hardening-sprint-verification.md)
