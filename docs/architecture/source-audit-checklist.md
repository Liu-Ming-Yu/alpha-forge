# Source Audit Checklist

Use this checklist for production-hardening reviews, bug investigations, and
architecture cleanup. Findings should become test-backed work items before any
live increment.

## Service Checks

- `services/data_service`: point-in-time access, stale-data fail-closed behavior,
  vendor fallback, dataset quorum, corporate-action handling.
- `services/research_service`: no lookahead leakage, governed feature admission,
  walk-forward evidence, calibrated costs, artifact integrity.
- `services/signal_service`: stable feature schema, bounded confidence,
  source attribution, promotion caps.
- `services/portfolio_service`: T+1 cash rules, reservations, ADV/sector/single
  name limits, drawdown and halted-instrument gates.
- `services/execution_service`: idempotent submission, reconnect, order
  uncertainty, reconciliation, kill-switch attribution.
- `services/governance_service`: readiness, paper soak, prediction quality,
  backup/restore, simulator calibration, operator auth.

## Runtime Surface Checks

- `bootstrap`: production profile settings fail closed before session creation.
- `engines`: shadow, paper, and live share data, regime, risk, and pre-trade
  gates.
- `infrastructure`: durable adapters hydrate restart-sensitive state and expose
  migration-backed contracts.
- `views/operator_api`: read models are authenticated, timeout-bounded, and
  explicit about unavailable evidence.
- `application`: use cases are typed, covered, and infrastructure-clean.
- `cli`: parsers bind typed request objects and avoid business logic.

## Required Evidence

- Full offline verification for production acceptance.
- Durable Postgres/Redis suite for durable-state changes.
- Live IBKR suite only when the change touches real broker integration.
- `mypy src` and full `pytest -q` for acceptance work.
- Architecture ratchets: import boundaries, service coupling, module size,
  type debt, lint debt, generated artifacts.
- Fresh prediction evidence for promoted non-classical sources.
- Feature-card and feature-audit evidence for governed feature sets.
- Paper-fill simulator calibration for execution-model promotion.
- Machine-owned paper-soak evidence before live promotion.

## Red Flags

- New direct cross-service import.
- New in-memory operator state without restart hydration.
- CLI/API handlers that submit orders or mutate transactional state directly.
- Live path that bypasses cash, risk, execution policy, or kill-switch checks.
- Research artifact compared across incompatible universes, horizons, or date
  policies.
- Any committed secret, broker identifier, generated Parquet, or cache artifact.
