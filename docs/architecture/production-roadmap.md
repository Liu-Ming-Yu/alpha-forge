# Production Roadmap

This roadmap records the current productionization state. It is not a promise
that live trading is safe; live promotion still requires current evidence and
operator approval.

## Phase 1: Runtime Foundation - Complete

Implemented:

- Modular monolith package layout.
- Core domain models and protocol contracts.
- Paper/live session factories.
- Cash-account first portfolio and execution path.
- Broker health checks and startup schema checks.
- Operator CLI command registry.

## Phase 2: Classical Alpha Pipeline - Complete

Implemented:

- Feature pipeline and cross-sectional factors.
- Linear signal scoring.
- Market-regime detector.
- Long-only and volatility-targeted constructors.
- Shared backtest/paper/live policy surfaces.

## Phase 3: Durable Persistence - Complete

Implemented:

- PostgreSQL repositories for orders, positions, audit, feature vectors, model
  registry, performance, readiness, and V2 state.
- Canonical Alembic tree under `src/quant_platform/alembic`.
- Startup and migration checks that fail closed on schema mismatch.

## Phase 3.5: Governance And Observability - Complete

Implemented:

- Preflight, data-health, performance, signal-gate, text-gate, readiness, and
  production-candidate commands.
- Prometheus metrics for cycle phases, order submit/reject, reconciliation,
  event bus, locks, and latency.
- Operator API authentication and rate limiting.
- Secrets and generated-artifact checks.

## Phase 4: Data Productionization - Complete Baseline

Implemented:

- Daily ingest and backfill.
- Intraday import and validation utilities.
- Corporate-action reprocessing.
- Vendor fallback and dataset quorum evidence.
- Maintenance loop and feature jobs.

## Phase 5: Text/LLM Feature Layer - Shadow/Gated

Implemented:

- Text event domain and stores.
- LLM text feature extraction.
- Text model manifest checks.
- Forecast evidence and startup assertion requirements.
- Shadow scoring and promotion gates.
- **`text-event-v2` feature family (27 features across three document
  kinds — news, SEC filings, earnings-call transcripts).** Tagged-union
  `ExtractedRecord` over `NewsExtraction` / `FilingExtraction` /
  `EarningsCallExtraction`, three versioned prompts
  (`news-prompt-v1` / `filing-prompt-v1` / `earnings-call-prompt-v1`),
  three per-kind aggregator panels with a shared core. Filings and
  earnings-call features are sparse-on-publication-date; news features
  are daily. All 27 ship evidence-gated
  (`expected_direction="unknown"`). Read-compatible with `v1` JSONL.
  See [`docs/text-event-v2-family.md`](../text-event-v2-family.md) for
  the full reference.

Live text influence remains gated by production-candidate evidence and a fresh
LLM live startup assertion.

## Phase 5.5: XGBoost Boosted Alpha Layer - Shadow/Gated

Implemented:

- XGBoost pairwise ranker.
- GPU diagnostics and CPU fallback.
- Manifest validation and model-registry support.
- Shadow/paper scoring behind feature-audit, signal-gate, and
  production-candidate evidence.

## Feature Factory — Registered Families

Parallel to the production-readiness phases above, the research-service
feature factory ships versioned families behind a single
`FamilyRegistry` contract. Each family registers via its package
`__init__.py`; `bootstrap_default_families()` materialises the whole set
at process start.

| Family | Version | Inputs | Count | Reference |
|---|---|---|---:|---|
| `price_volume` | `price-volume-starter-v1` | Daily OHLCV | 27 | `research/features/price_volume/` |
| `fundamentals` | `fundamentals-plus-v1` | Sharadar SF1 ARQ | 41 | `research/features/fundamentals/` |
| `formulaic` | `formulaic-v1` (+ auto-promoted) | Daily OHLCV | 9 curated + auto | `research/features/formulaic/` |
| `text` | `text-event-v2` | LLM extractions | 27 | [`text-event-v2-family.md`](../text-event-v2-family.md) |
| `microstructure` | `microstructure-v3` | Daily OHLCV | 19 | [`microstructure-v3-family.md`](../microstructure-v3-family.md) |
| `ownership` | `ownership-v1` | 13F + short-interest + shares-out records (scaffold) | 6 | [`ownership-v1-family.md`](../ownership-v1-family.md) |
| `estimates` | `estimates-v1` | Analyst consensus snapshots + surprise records (scaffold) | 6 | [`estimates-v1-family.md`](../estimates-v1-family.md) |
| `options` | `options-v1` | Options snapshots (scaffold) | 6 | [`options-v1-family.md`](../options-v1-family.md) |
| `macro` | `macro-v1` | FRED macro time series (scaffold + optional FRED helper) | 6 | [`macro-v1-family.md`](../macro-v1-family.md) |
| `learned` | `learned-representations-v1` | The other 9 families' outputs + frozen PCA artifact | 9 | [`learned-representations-v1-family.md`](../learned-representations-v1-family.md) |

Total: **156 features** across ten families. All families enforce
`expected_direction="unknown"` for new specs; direction promotion is a
family-version bump, not an in-place edit. The walk-forward + signal-gate
pipeline is what earns a feature its direction.

## Phase 6: Scale Controls And Advanced Execution - Started

Implemented foundation:

- V2 account orchestration.
- Multi-engine proposals and budgets.
- Dataset quorum evidence.
- Participation-aware simulator.
- Simulator calibration artifact.
- Execution-quality evidence.

Remaining work:

- Deeper calibration against paper fill distributions.
- More tactic-specific execution evidence.
- Further durable telemetry and operator action rollups.
- Continued module-size, lint, and type ratchets.

## Deferred Scope

- Multi-broker support.
- Margin, shorts, options, futures, leverage.
- Separate deployable microservices.
- Full order-book simulator.
- HFT or sub-second execution strategies.
- General multi-tenant operator model.
