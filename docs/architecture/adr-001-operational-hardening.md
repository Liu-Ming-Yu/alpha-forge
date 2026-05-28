# ADR-001: Operational production-readiness ("Track 1")

**Status:** Accepted — implementation complete (items 1-7 shipped 2026-05-24 → 2026-05-26).
**Date:** 2026-05-24
**Deciders:** Operator

> **Status update 2026-05-27** — All seven items in this ADR have landed.
> ADR-003 subsequently shipped a corrected backtest accounting that
> changes one factual claim in this Context section: the "every
> evaluated signal class has IC ≈ 0" framing was an artefact of the
> v1 return-accounting bug compounding a 21d label as 1d P&L; under
> the corrected v2 accounting, the latest stack shows positive IC
> (Arm A bootstrap CI roughly [-0.0002, 0.0159], Arm A passes the
> Sharpe gate at 1.05). Eligibility still fails on
> `fold_negative_ic_streak`, so the data-blocked direction is
> roughly right but the magnitudes were wrong. The operational
> hardening below remains valid regardless. See
> [ADR-003](adr-003-return-accounting-separation.md).

## Context

After 7 alpha-improvement workstreams plus walk-forward XGBoost (PR #25) and the
full LLM SEC-text pipeline (PR #26), every evaluated signal class — linear,
gradient-boosted, LLM-text — has IC ≈ 0 on liquid US large-caps 2023–2026 *under
the v1 backtest accounting in force at the time of writing* (see status update
above for the v2-corrected picture). The governance gates correctly refuse to
promote. *Strategy* production-readiness is therefore data-blocked (orthogonal
signal needed: news, fundamentals, or a less-arbitraged universe — see
`docs/text-event-alpha-scope.md`).

The *platform itself* is mature but has accumulated recurring papercuts that
will bite the moment a signal does pass: parquet duplicate rows from re-ingest,
silent extractor defaults that don't match deployed prompts, Docker drops on
Windows wiping running jobs, no automated backups, no alerting beyond stdout
logs, no DR procedure, secrets in `.env` with no rotation.

This ADR scopes the operational hardening — the work that's worth doing
regardless of whether a signal exists, because the day one does the stack must
be trustworthy.

## Decision

Execute a 7-item operational hardening pass on branch `text-event-alpha`
(merged via PR #26 or a follow-up). Treat it as ops-only — no alpha-discovery
work mixed in. Respect platform invariants (cli → application → services →
core; no service-to-service imports; fail-closed defaults; module ≤300 lines).

## Options considered

### Option A: Bundle every item into one large PR
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Review time | Long — reviewer must context-switch across 7 areas |
| Risk | Higher — one item breaking blocks all the others |
| Cost | Operator-time-only |

**Pros:** Single artifact, one CI run. **Cons:** harder to revert, larger blast
radius, slows merge.

### Option B: Per-item PRs, sequenced
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low per PR, but more total PRs |
| Review time | Fast individually |
| Risk | Lowest — each item is isolable |
| Cost | More PR overhead |

**Pros:** Each fix is reviewable in minutes. **Cons:** 7 PRs is noisy.

### Option C: Two grouped PRs — "papercuts" (small fixes) + "ops tooling" (new code/docs)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Review time | Two focused PRs, each scoped |
| Risk | Low — natural boundary between fixes and new components |
| Cost | Minimal |

**Pros:** Logical grouping, each PR has one purpose. **Cons:** none material.

## Trade-off analysis

Option C wins. The 7 items split cleanly into:
- **Fixes** (items 2–4): one-or-two-line edits that don't introduce new
  components, only correct existing ones.
- **Tooling + docs** (items 5–7): new scripts, runbooks, and an alert wiring
  that introduce new conventions (covered by `platform-conventions.md`).

Item 1 (merge open PRs) is independent of this branch.

## Decision: scope and sequencing

| # | Item | Type | Effort | Files |
|---|------|------|--------|-------|
| 1 | Merge PRs #25 and #26 | external | — | n/a (operator) |
| 2 | Postgres healthcheck `pg_isready -d quant_platform` | fix | 1 line | `docker-compose.yml` |
| 3 | Parquet bar upsert (dedup-on-write) | fix | ~20 lines | `services/data_service/bar_store/...` |
| 4 | LLM defaults (`max_tokens=3000`, `timeout=120`, `max_request_latency=120`) into `LLMSettings`. Platform invariant `max_request_latency <= timeout` is enforced by `services/governance_service/llm_live_startup/runtime_limits.py` — both knobs raised together. | fix | ~6 lines | `config_signal_models/llm.py`, `infra/config/settings.example.env` |
| 5 | Daily backup script + cron snippet | tooling | new script | `scripts/backup_durable.py`, runbook |
| 6 | Supervise watchdog + alerting playbook | docs + small script | runbook + sample | `docs/runbooks/paper-soak-watchdog.md`, `scripts/alert_failures.py` |
| 7 | Disaster-recovery doc + secrets rotation plan | docs | runbook pair | `docs/runbooks/disaster-recovery.md`, `docs/runbooks/secrets-rotation.md` |

## Consequences

**Easier:**
- Future re-ingest will not silently duplicate parquet rows.
- Operator can leave the supervise loop running for 24h+ without losing context
  when Docker hiccups (auto-restart + watchdog + alert).
- LLM extraction starts with the *correct* defaults, not the truncated-JSON /
  retry-storm defaults the platform shipped with.
- A signal that passes audit promotes through a stack with tested backups and
  a documented DR procedure.

**Harder:**
- Slightly more YAML and a couple of cron jobs to maintain.
- Secrets rotation requires a discipline the platform didn't enforce before.

**Revisit when:**
- A signal genuinely passes eligibility — at that point the supervise loop and
  alerting become live-critical and may need tightening.
- Docker Desktop instability persists — consider migrating durable services to
  WSL2-native processes or a small VPS to remove the Windows GUI dependency.

## Action items

1. [x] (operator) Merge PR #25 and PR #26 once green. — PR #25 merged 2026-05-22, PR #26 merged 2026-05-24.
2. [x] Postgres healthcheck `pg_isready -d quant_platform`.
3. [x] Parquet bar dedup-on-write.
4. [x] LLM defaults into `LLMSettings` + example env.
5. [x] `scripts/backup_durable.py` + runbook.
6. [x] Supervise watchdog runbook + `scripts/alert_failures.py`.
7. [x] `docs/runbooks/disaster-recovery.md`, `docs/runbooks/secrets-rotation.md`.

All items complete. The "revisit when" triggers — a passing signal or
sustained Docker Desktop instability — have not fired yet; this ADR
stays Accepted but its operational items are no longer active work.
