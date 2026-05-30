# ADR-014 — Unified engine runtime (single-engine as the N=1 multi-engine case)

**Status:** Accepted (incremental). The supervised production loop and the bounded `run-multi-engine` path now execute through **one** runner — `AccountOrchestratorLoopRunner` — driven by the existing robust loop (`run_engine_loop`). A single engine is just one engine + one auto-budget, so it writes the same governance/read-model rows (`strategy_runs`, `engine_budgets`, `combined_portfolio_targets`, `engine_target_contributions`) the operator console reads. Verified: `supervise --max-cycles 2` (V2 enabled) completes through the loop and the console reports the run. The global `QP__V2__ENABLED` default is **deliberately not** flipped (see Consequences); V2 is enabled via config/`.env`.

## Context

The platform had **two write paths that fed one read model**:

- **V1 single-engine** — `run-cycle` / `run-engine` / `supervise` (`run_engine_loop` → `EngineRunner.run_cycle`). Robust loop (kill-switch refresh, recovery assessment, interval sleep, signal handlers, per-cycle error isolation), but it wrote only `feature_vectors`.
- **V2 multi-engine** — `run-multi-engine` (`AccountExecutionOrchestrator` + `MultiEngineRunner` + governance repo). Wrote the governance/read-model tables, but its cycle loop was a bare `for` with none of the loop robustness.

The operator console **Strategy** screen (strategy runs, ensemble blend weights, source contributions, engine budgets) reads tables written **only** by the V2 path. So running the V1 path — the actual production reality (Arm G is a single strategy) — left the console empty by construction. This divergence also produced a string of latent bugs (orphan feature jobs halting cycles; a budget-key/proposal-name mismatch; `strategy_runs` had no writer at all; an out-of-range auto-budget `max_gross`).

There was no ADR for the V1/V2 split — it accreted (the `008_multi_engine_governance` migration), it was never a decision.

Constraints: single-operator modular monolith; fail-closed execution; durable Postgres; the change touches the most safety-critical code (the execution loop), so it must preserve the loop's robustness exactly and be landable incrementally.

## Decision

**1. One execution path via an orchestrator-backed loop runner.** `run_engine_loop` already drives an injectable `runner_factory` producing an `EngineLoopRunner` (`initialize` / `run_cycle → CycleResult` / `shutdown`) and pulls a session via `engine_session()` for kill-switch/recovery. We add `AccountOrchestratorLoopRunner` (in `bootstrap/engine/orchestrator_runner.py`) implementing that protocol: `initialize` assembles the proposal engine(s), execution session, V2 repo bundle, governance repo, `MultiEngineRunner`, and `AccountExecutionOrchestrator`; `run_cycle` runs one orchestrator cycle (sync account → prices → proposals → **persist run** → execute → mark run completed) and returns a `CycleResult`. It exposes `_session` so the loop's kill-switch/recovery work unchanged.

**2. Single-engine is N=1.** `supervise` injects this runner (auto-budget, `capital_weight=1.0`) when `v2.enabled and v2.account_orchestrator_enabled`, else falls back to the V1 runner. `run-multi-engine` now drives the **same** runner via a thin bounded loop. The per-cycle execution lives in exactly one place, so the two entrypoints can never diverge again.

**3. One engine identity.** Budgets are keyed by the plugin's canonical proposal name (`cross_sectional_equity_v1`), not the CLI/registry key (`cross_sectional_equity`), via `_canonical_engine_name` — applied in both the file and auto-budget paths. This removes the class of budget/proposal lookup mismatches.

**4. Do not flip the global V2 default.** Enabling V2 by config is correct; flipping the library default (`V2Settings.enabled = True`) is **not** done here. Evidence: enabling V2 in the operator's `.env` already leaked into several poorly-isolated tests (they read `.env` even via `model_validate`) and would fail V2 wiring without a DSN. A global default flip would hit every such test. Instead: V2 is enabled per-deployment via `.env`/config, and a root `tests/conftest.py` pins the V2 flags off for the suite (env precedence over `.env`; V2 tests set `v2=V2Settings(enabled=True)` explicitly).

## Consequences

- **Positive:** The console reflects whatever runs. One write path → one read model → no recurring divergence. The robust loop (kill-switch, recovery, interval, signals, error isolation) now backs multi-engine too. Latent bugs fixed at the source (orphan feature jobs, budget naming, missing `strategy_runs` writer, auto-budget `max_gross`).
- **Negative / follow-ups:** `run-cycle` (the dev single-cycle helper) still uses the V1 path — lower priority, returns a `CycleResult` not an `EngineLoopSummary`. `CycleResult.signals` is empty on the orchestrator path (it works at the proposal/target level), so the `engine_loop.cycle_complete` log shows `signals: 0` — cosmetic; the console reads signals from governance, not `CycleResult`. The global-default flip and full test-isolation hardening remain deferred.
- **Architecture:** The new `bootstrap → engines` composition edges for `orchestrator_runner.py` are registered, and the now-removed `multi.py → engines.account.orchestrator`/`multi_engine` approvals are dropped from the ratchet.
- **Revisit when:** true multi-account / capital-allocation scale arrives — that is the trigger to move run-state transitions onto the existing Redis Streams bus and introduce an event-projected read model (the previously-considered Option D), which the orchestrator's budget seam already anticipates.
